use std::collections::HashMap;
use std::io::Cursor;
use std::{any::Any, sync::Arc};

use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow_ipc::reader::{StreamDecoder, StreamReader};
use async_trait::async_trait;
use clickhouse::{Client, Row};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::EmptyRecordBatchStream;
use datafusion::{execution::SendableRecordBatchStream, sql::TableReference};
use regex::Regex;
use serde::Deserialize;
use snafu::ResultExt;

use super::{AsyncDbConnection, DbConnection, Error, SyncDbConnection};

impl DbConnection<Client, ()> for Client {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_sync(&self) -> Option<&dyn SyncDbConnection<Client, ()>> {
        None
    }

    fn as_async(&self) -> Option<&dyn AsyncDbConnection<Client, ()>> {
        Some(self)
    }
}

#[async_trait]
impl AsyncDbConnection<Client, ()> for Client {
    fn new(conn: Client) -> Self
    where
        Self: Sized,
    {
        conn
    }

    async fn tables(&self, schema: &str) -> Result<Vec<String>, Error> {
        #[derive(Row, Deserialize)]
        struct Row {
            name: String,
        }

        let tables: Vec<Row> = self
            .query("SELECT name FROM system.tables WHERE database = ?")
            .bind(schema)
            .fetch_all()
            .await
            .boxed()
            .context(super::UnableToGetTablesSnafu)?;

        Ok(tables.into_iter().map(|x| x.name).collect())
    }

    async fn schemas(&self) -> Result<Vec<String>, Error> {
        #[derive(Row, Deserialize)]
        struct Row {
            name: String,
        }
        let tables: Vec<Row> = self
            .query("SELECT name FROM system.databases WHERE name NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')")
            .fetch_all()
            .await
            .boxed()
            .context(super::UnableToGetSchemasSnafu)?;

        Ok(tables.into_iter().map(|x| x.name).collect())
    }

    /// Get the schema for a table reference.
    ///
    /// # Arguments
    ///
    /// * `table_reference` - The table reference.
    async fn get_schema(&self, table_reference: &TableReference) -> Result<SchemaRef, Error> {
        #[derive(Row, Deserialize)]
        struct CatalogRow {
            db: String,
        }

        let database = match table_reference.schema() {
            Some(db) => db.to_string(),
            None => {
                let row: CatalogRow = self
                    .query("SELECT currentDatabase() AS db")
                    .fetch_one()
                    .await
                    .boxed()
                    .context(super::UnableToGetSchemaSnafu)?;
                row.db
            }
        };

        #[derive(Row, Deserialize)]
        struct TableInfoRow {
            engine: String,
            as_select: String,
        }

        let table_info: TableInfoRow = self
            .query("SELECT engine, as_select FROM system.tables WHERE database = ? AND name = ?")
            .bind(&database)
            .bind(table_reference.table())
            .fetch_one()
            .await
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        let is_view = matches!(
            table_info.engine.to_uppercase().as_str(),
            "VIEW" | "MATERIALIZEDVIEW"
        );

        let statement = if is_view {
            let view_query = table_info.as_select;
            format!(
                "SELECT * FROM ({}) LIMIT 0",
                replace_clickhouse_ddl_parameters(&view_query)
            )
        } else {
            let table_ref = TableReference::partial(database.clone(), table_reference.table());
            format!("SELECT * FROM {} LIMIT 0", table_ref.to_quoted_string())
        };

        let mut bytes = self
            .query(&statement)
            .fetch_bytes("ArrowStream")
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        let reader = bytes
            .collect()
            .await
            .boxed()
            .and_then(|bytes| StreamReader::try_new(Cursor::new(bytes), None).boxed())
            .context(super::UnableToGetSchemaSnafu)?;

        let arrow_schema = reader.schema();

        // Query the true ClickHouse types from system.columns and remap the schema.
        // ClickHouse's ArrowStream format returns Date as UInt16 and DateTime as UInt32,
        // but DataFusion needs proper Date32/Timestamp types.
        let ch_types = get_clickhouse_column_types(self, &database, table_reference.table()).await?;
        let remapped_schema = remap_schema_with_clickhouse_types(&arrow_schema, &ch_types);

        Ok(remapped_schema)
    }

    /// Query the database with the given SQL statement and parameters, returning a `Result` of `SendableRecordBatchStream`.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement.
    /// * `params` - The parameters for the SQL statement.
    /// * `projected_schema` - The Projected schema for the query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    async fn query_arrow(
        &self,
        sql: &str,
        _params: &[()],
        projected_schema: Option<SchemaRef>,
    ) -> super::Result<SendableRecordBatchStream> {
        let query = self.query(sql);

        let mut bytes_stream = query
            .fetch_bytes("ArrowStream")
            .boxed()
            .context(super::UnableToQueryArrowSnafu)?;

        let mut first_batch: Option<RecordBatch> = None;
        let mut decoder = StreamDecoder::new();

        // fetch till first set of records
        while let Some(buf) = bytes_stream.next().await? {
            if let Some(batch) = decoder.decode(&mut buf.into())? {
                first_batch = Some(batch);
                break;
            }
        }

        if let Some(first_batch) = first_batch {
            // Use projected_schema if provided, otherwise use the batch's schema.
            // ClickHouse may return data with a different schema than what was
            // declared at planning time (e.g., DateTime as UInt32). We need to
            // ensure the returned batches match the expected schema.
            let target_schema = projected_schema.unwrap_or_else(|| first_batch.schema());
            let first_batch = cast_batch_to_schema(&first_batch, &target_schema)?;

            let stream_schema = Arc::clone(&target_schema);
            let stream = async_stream::stream! {
                yield Ok(first_batch);
                while let Some(buf) = bytes_stream
                    .next()
                    .await
                    .map_err(|er| arrow::error::ArrowError::ExternalError(Box::new(er)))?
                {
                    if let Some(batch) = decoder.decode(&mut buf.into())? {
                        let batch = cast_batch_to_schema(&batch, &stream_schema)?;
                        yield Ok(batch);
                    }
                }
            };
            Ok(Box::pin(RecordBatchStreamAdapter::new(target_schema, stream)))
        } else if let Some(schema) = projected_schema {
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                EmptyRecordBatchStream::new(schema),
            )))
        } else {
            let schema: Arc<Schema> = Schema::empty().into();
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                EmptyRecordBatchStream::new(schema),
            )))
        }
    }

    /// Execute the given SQL statement with parameters, returning the number of affected rows.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement.
    /// * `params` - The parameters for the SQL statement.
    async fn execute(&self, sql: &str, params: &[()]) -> super::Result<u64> {
        let mut query = self.query(sql);

        for param in params {
            query = query.bind(param);
        }

        query
            .execute()
            .await
            .boxed()
            .context(super::UnableToQueryArrowSnafu)?;

        Ok(0)
    }
}

/// Query the true ClickHouse column types from system.columns.
async fn get_clickhouse_column_types(
    client: &Client,
    database: &str,
    table: &str,
) -> Result<HashMap<String, String>, Error> {
    #[derive(Row, Deserialize)]
    struct ColumnTypeRow {
        name: String,
        #[serde(rename = "type")]
        col_type: String,
    }

    let rows: Vec<ColumnTypeRow> = client
        .query("SELECT name, type FROM system.columns WHERE database = ? AND table = ?")
        .bind(database)
        .bind(table)
        .fetch_all()
        .await
        .boxed()
        .context(super::UnableToGetSchemaSnafu)?;

    Ok(rows.into_iter().map(|r| (r.name, r.col_type)).collect())
}

/// Remap an Arrow schema based on the true ClickHouse column types.
///
/// ClickHouse's ArrowStream format returns temporal types as integers:
/// - Date → UInt16 (days since 1970-01-01)
/// - DateTime → UInt32 (seconds since epoch)
/// - DateTime64(N) → Int64 (scaled time units)
///
/// This function remaps these to proper Arrow temporal types so DataFusion's
/// date/time functions work correctly.
fn remap_schema_with_clickhouse_types(
    arrow_schema: &SchemaRef,
    ch_types: &HashMap<String, String>,
) -> SchemaRef {
    let new_fields: Vec<Field> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            let field_name = field.name();
            if let Some(ch_type) = ch_types.get(field_name) {
                if let Some(arrow_type) = clickhouse_type_to_arrow(ch_type) {
                    return Field::new(field_name, arrow_type, field.is_nullable());
                }
            }
            field.as_ref().clone()
        })
        .collect();

    Arc::new(Schema::new_with_metadata(
        new_fields,
        arrow_schema.metadata().clone(),
    ))
}

/// Convert a ClickHouse type string to the appropriate Arrow DataType.
///
/// Only converts types that ClickHouse serializes incorrectly in ArrowStream:
/// - Date → Date32
/// - DateTime → Timestamp(Second, tz)
/// - DateTime64(N) → Timestamp with appropriate precision
fn clickhouse_type_to_arrow(ch_type: &str) -> Option<DataType> {
    let ch_type_trimmed = ch_type.trim();

    // Handle Nullable wrapper
    if ch_type_trimmed.starts_with("Nullable(") && ch_type_trimmed.ends_with(')') {
        let inner = &ch_type_trimmed[9..ch_type_trimmed.len() - 1];
        return clickhouse_type_to_arrow(inner);
    }

    // Handle LowCardinality wrapper
    if ch_type_trimmed.starts_with("LowCardinality(") && ch_type_trimmed.ends_with(')') {
        let inner = &ch_type_trimmed[15..ch_type_trimmed.len() - 1];
        return clickhouse_type_to_arrow(inner);
    }

    // Date types
    if ch_type_trimmed == "Date" {
        return Some(DataType::Date32);
    }
    if ch_type_trimmed == "Date32" {
        return Some(DataType::Date32);
    }

    // DateTime without timezone: DateTime
    if ch_type_trimmed == "DateTime" {
        return Some(DataType::Timestamp(TimeUnit::Second, None));
    }

    // DateTime with timezone: DateTime('UTC') or DateTime('America/New_York')
    if ch_type_trimmed.starts_with("DateTime(") && ch_type_trimmed.ends_with(')') {
        let inner = &ch_type_trimmed[9..ch_type_trimmed.len() - 1];
        // Remove quotes from timezone
        let tz = inner.trim_matches('\'').trim_matches('"');
        if tz.is_empty() {
            return Some(DataType::Timestamp(TimeUnit::Second, None));
        }
        return Some(DataType::Timestamp(
            TimeUnit::Second,
            Some(tz.to_string().into()),
        ));
    }

    // DateTime64 with precision: DateTime64(3), DateTime64(6), DateTime64(9)
    // Optionally with timezone: DateTime64(3, 'UTC')
    if ch_type_trimmed.starts_with("DateTime64(") && ch_type_trimmed.ends_with(')') {
        let inner = &ch_type_trimmed[11..ch_type_trimmed.len() - 1];
        let parts: Vec<&str> = inner.splitn(2, ',').collect();

        let precision: u32 = parts[0].trim().parse().unwrap_or(9);
        let tz = if parts.len() > 1 {
            let tz_str = parts[1].trim().trim_matches('\'').trim_matches('"').trim();
            if tz_str.is_empty() {
                None
            } else {
                Some(tz_str.to_string().into())
            }
        } else {
            None
        };

        let time_unit = match precision {
            0 => TimeUnit::Second,
            1..=3 => TimeUnit::Millisecond,
            4..=6 => TimeUnit::Microsecond,
            _ => TimeUnit::Nanosecond,
        };

        return Some(DataType::Timestamp(time_unit, tz));
    }

    // Not a type we need to remap
    None
}

pub fn replace_clickhouse_ddl_parameters(ddl_query: &str) -> String {
    // Regex to find parameters in the format {parameter_name:DataType}
    let param_pattern = Regex::new(r"\{(\w+?):(\w+?)\}").unwrap();

    let modified_query = param_pattern.replace_all(ddl_query, |caps: &regex::Captures| {
        // match against the datatype
        let data_type = caps.get(2).map_or("", |m| m.as_str());
        match data_type.to_lowercase().as_str() {
            "string" => "''".to_string(),
            "uint8" | "uint16" | "uint32" | "uint64" | "int8" | "int16" | "int32" | "int64" => {
                "0".to_string()
            }
            "float32" | "float64" => "0.0".to_string(),
            "date" => "'2000-01-01'".to_string(),
            "datetime" => "'2000-01-01 00:00:00'".to_string(),
            "bool" => "false".to_string(),
            _ => "''".to_string(),
        }
    });

    modified_query.into_owned()
}

/// Cast a RecordBatch to match a target schema.
///
/// ClickHouse may return data with types that differ from the schema obtained at
/// planning time (e.g., DateTime columns may be returned as UInt32). This function
/// casts each column to match the expected type in the target schema.
///
/// Special handling for ClickHouse temporal types:
/// - UInt16 → Date32: ClickHouse sends Date as UInt16 (days since epoch)
/// - UInt32 → Timestamp(Second, _): ClickHouse sends DateTime as UInt32 (seconds since epoch)
/// - Int64 → Timestamp(_, _): ClickHouse sends DateTime64 as Int64
///
/// If schemas already match, the batch is returned unchanged.
fn cast_batch_to_schema(
    batch: &RecordBatch,
    target_schema: &SchemaRef,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    use arrow::array::{Array, Date32Array, PrimitiveArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray};
    use arrow::datatypes::{Int64Type, UInt16Type, UInt32Type};

    let batch_schema = batch.schema();

    // Quick path: if schemas match, return as-is
    if batch_schema == *target_schema {
        return Ok(batch.clone());
    }

    // Cast each column to the target type
    let columns: Result<Vec<_>, _> = batch
        .columns()
        .iter()
        .zip(target_schema.fields())
        .map(|(col, target_field)| {
            if col.data_type() == target_field.data_type() {
                return Ok(Arc::clone(col));
            }

            // Handle special ClickHouse temporal type conversions
            match (col.data_type(), target_field.data_type()) {
                // UInt16 → Date32: ClickHouse Date is days since 1970-01-01
                (DataType::UInt16, DataType::Date32) => {
                    let uint16_array = col.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>()
                        .ok_or_else(|| arrow::error::ArrowError::CastError(
                            "Expected UInt16Array for Date conversion".to_string()
                        ))?;
                    // UInt16 values are days since epoch, which is exactly what Date32 expects
                    let date_array: Date32Array = uint16_array
                        .iter()
                        .map(|v| v.map(|d| d as i32))
                        .collect();
                    Ok(Arc::new(date_array) as Arc<dyn Array>)
                }

                // UInt32 → Timestamp(Second, tz): ClickHouse DateTime is seconds since epoch
                (DataType::UInt32, DataType::Timestamp(TimeUnit::Second, tz)) => {
                    let uint32_array = col.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>()
                        .ok_or_else(|| arrow::error::ArrowError::CastError(
                            "Expected UInt32Array for DateTime conversion".to_string()
                        ))?;
                    let ts_array: TimestampSecondArray = uint32_array
                        .iter()
                        .map(|v| v.map(|ts| ts as i64))
                        .collect::<TimestampSecondArray>()
                        .with_timezone_opt(tz.clone());
                    Ok(Arc::new(ts_array) as Arc<dyn Array>)
                }

                // Int64 → Timestamp with various precisions (DateTime64)
                (DataType::Int64, DataType::Timestamp(unit, tz)) => {
                    let int64_array = col.as_any().downcast_ref::<PrimitiveArray<Int64Type>>()
                        .ok_or_else(|| arrow::error::ArrowError::CastError(
                            "Expected Int64Array for DateTime64 conversion".to_string()
                        ))?;
                    match unit {
                        TimeUnit::Second => {
                            let ts_array: TimestampSecondArray = int64_array
                                .iter()
                                .collect::<TimestampSecondArray>()
                                .with_timezone_opt(tz.clone());
                            Ok(Arc::new(ts_array) as Arc<dyn Array>)
                        }
                        TimeUnit::Millisecond => {
                            let ts_array: TimestampMillisecondArray = int64_array
                                .iter()
                                .collect::<TimestampMillisecondArray>()
                                .with_timezone_opt(tz.clone());
                            Ok(Arc::new(ts_array) as Arc<dyn Array>)
                        }
                        TimeUnit::Microsecond => {
                            let ts_array: TimestampMicrosecondArray = int64_array
                                .iter()
                                .collect::<TimestampMicrosecondArray>()
                                .with_timezone_opt(tz.clone());
                            Ok(Arc::new(ts_array) as Arc<dyn Array>)
                        }
                        TimeUnit::Nanosecond => {
                            let ts_array: TimestampNanosecondArray = int64_array
                                .iter()
                                .collect::<TimestampNanosecondArray>()
                                .with_timezone_opt(tz.clone());
                            Ok(Arc::new(ts_array) as Arc<dyn Array>)
                        }
                    }
                }

                // Default: use Arrow's cast function
                _ => cast(col, target_field.data_type()),
            }
        })
        .collect();

    RecordBatch::try_new(Arc::clone(target_schema), columns?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, Int64Array, TimestampSecondArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    #[test]
    fn test_cast_batch_to_schema_same_schema() {
        // When schemas match, the batch should be returned as-is
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let result = cast_batch_to_schema(&batch, &schema).unwrap();
        assert_eq!(result.schema(), schema);
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_cast_batch_to_schema_uint32_to_timestamp() {
        // This is the actual bug case: ClickHouse returns DateTime as UInt32
        // but schema expects Timestamp
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("ts", DataType::UInt32, false),
        ]));

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), false),
        ]));

        let batch = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(UInt32Array::from(vec![1609459200, 1609545600])), // Unix timestamps
            ],
        )
        .unwrap();

        let result = cast_batch_to_schema(&batch, &target_schema).unwrap();

        assert_eq!(result.schema(), target_schema);
        assert_eq!(result.num_rows(), 2);

        // Verify the timestamp values were correctly cast
        let ts_col = result
            .column(1)
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .unwrap();
        assert_eq!(ts_col.value(0), 1609459200);
        assert_eq!(ts_col.value(1), 1609545600);
    }

    #[test]
    fn test_cast_batch_to_schema_int32_to_int64() {
        // Test widening integer cast
        let source_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));

        let target_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));

        let batch = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(Int32Array::from(vec![100, 200, 300]))],
        )
        .unwrap();

        let result = cast_batch_to_schema(&batch, &target_schema).unwrap();

        assert_eq!(result.schema(), target_schema);
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 100);
        assert_eq!(col.value(1), 200);
        assert_eq!(col.value(2), 300);
    }

    #[test]
    fn test_cast_batch_to_schema_partial_cast() {
        // Only some columns need casting
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![10, 20])),
            ],
        )
        .unwrap();

        let result = cast_batch_to_schema(&batch, &target_schema).unwrap();

        assert_eq!(result.schema(), target_schema);

        // First column should be unchanged (same type)
        let id_col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 1);

        // Second column should be cast
        let value_col = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(value_col.value(0), 10);
    }

    #[test]
    fn test_replace_clickhouse_ddl_parameters() {
        // Test existing functionality
        let query = "SELECT * FROM table WHERE id = {id:Int64} AND name = {name:String}";
        let result = replace_clickhouse_ddl_parameters(query);
        assert_eq!(
            result,
            "SELECT * FROM table WHERE id = 0 AND name = ''"
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_date() {
        assert_eq!(
            clickhouse_type_to_arrow("Date"),
            Some(DataType::Date32)
        );
        assert_eq!(
            clickhouse_type_to_arrow("Date32"),
            Some(DataType::Date32)
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_datetime() {
        assert_eq!(
            clickhouse_type_to_arrow("DateTime"),
            Some(DataType::Timestamp(TimeUnit::Second, None))
        );
        assert_eq!(
            clickhouse_type_to_arrow("DateTime('UTC')"),
            Some(DataType::Timestamp(TimeUnit::Second, Some("UTC".into())))
        );
        assert_eq!(
            clickhouse_type_to_arrow("DateTime('America/New_York')"),
            Some(DataType::Timestamp(TimeUnit::Second, Some("America/New_York".into())))
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_datetime64() {
        // DateTime64 with different precisions
        assert_eq!(
            clickhouse_type_to_arrow("DateTime64(0)"),
            Some(DataType::Timestamp(TimeUnit::Second, None))
        );
        assert_eq!(
            clickhouse_type_to_arrow("DateTime64(3)"),
            Some(DataType::Timestamp(TimeUnit::Millisecond, None))
        );
        assert_eq!(
            clickhouse_type_to_arrow("DateTime64(6)"),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
        assert_eq!(
            clickhouse_type_to_arrow("DateTime64(9)"),
            Some(DataType::Timestamp(TimeUnit::Nanosecond, None))
        );
        // With timezone
        assert_eq!(
            clickhouse_type_to_arrow("DateTime64(3, 'UTC')"),
            Some(DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())))
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_nullable() {
        assert_eq!(
            clickhouse_type_to_arrow("Nullable(Date)"),
            Some(DataType::Date32)
        );
        assert_eq!(
            clickhouse_type_to_arrow("Nullable(DateTime)"),
            Some(DataType::Timestamp(TimeUnit::Second, None))
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_lowcardinality() {
        assert_eq!(
            clickhouse_type_to_arrow("LowCardinality(Nullable(Date))"),
            Some(DataType::Date32)
        );
    }

    #[test]
    fn test_clickhouse_type_to_arrow_non_temporal() {
        // Non-temporal types should return None (not remapped)
        assert_eq!(clickhouse_type_to_arrow("String"), None);
        assert_eq!(clickhouse_type_to_arrow("UInt32"), None);
        assert_eq!(clickhouse_type_to_arrow("Int64"), None);
        assert_eq!(clickhouse_type_to_arrow("Float64"), None);
    }

    #[test]
    fn test_remap_schema_with_clickhouse_types() {
        use std::collections::HashMap;

        // Simulate a schema from ClickHouse ArrowStream (wrong types)
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("created_at", DataType::UInt16, false),      // Should be Date32
            Field::new("updated_at", DataType::UInt32, false),      // Should be Timestamp
            Field::new("name", DataType::Utf8, false),              // Should stay Utf8
        ]));

        let mut ch_types = HashMap::new();
        ch_types.insert("id".to_string(), "Int64".to_string());
        ch_types.insert("created_at".to_string(), "Date".to_string());
        ch_types.insert("updated_at".to_string(), "DateTime".to_string());
        ch_types.insert("name".to_string(), "String".to_string());

        let remapped = remap_schema_with_clickhouse_types(&arrow_schema, &ch_types);

        assert_eq!(remapped.field(0).data_type(), &DataType::Int64);
        assert_eq!(remapped.field(1).data_type(), &DataType::Date32);
        assert_eq!(remapped.field(2).data_type(), &DataType::Timestamp(TimeUnit::Second, None));
        assert_eq!(remapped.field(3).data_type(), &DataType::Utf8);
    }
}
