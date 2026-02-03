/*
Copyright 2024 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use clickhouse::Client;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::Result as DataFusionResult;
use datafusion::sql::sqlparser::ast::{
    self, Expr, FunctionArg, FunctionArgExpr, FunctionArgOperator, Ident, ObjectName, Value,
};
use datafusion::sql::sqlparser::tokenizer::Span;
use datafusion::sql::unparser;
use datafusion::sql::unparser::Unparser;
use datafusion::{common::Constraints, sql::TableReference};
use std::sync::Arc;

use crate::sql::db_connection_pool::clickhousepool::ClickHouseConnectionPool;
use crate::sql::db_connection_pool::dbconnection::AsyncDbConnection;

#[cfg(feature = "clickhouse-federation")]
mod federation;
mod sql_table;

/// ClickHouse-specific SQL dialect for unparsing DataFusion expressions.
///
/// This dialect translates DataFusion function names to their ClickHouse equivalents
/// when generating SQL to send to a ClickHouse backend.
pub struct ClickHouseDialect {}

impl ClickHouseDialect {
    pub fn new() -> Self {
        Self {}
    }

    /// Maps DataFusion function names to ClickHouse function names.
    fn get_function_name_mapping(func_name: &str) -> Option<&'static str> {
        match func_name {
            // Map functions (DataFusion uses snake_case, ClickHouse uses camelCase)
            "map_keys" => Some("mapKeys"),
            "map_values" => Some("mapValues"),
            // Array functions
            "array_has" => Some("has"),
            "array_has_all" => Some("hasAll"),
            "array_has_any" => Some("hasAny"),
            "array_length" | "array_ndims" => Some("length"),
            "array_concat" => Some("arrayConcat"),
            "array_distinct" => Some("arrayDistinct"),
            "array_element" => Some("arrayElement"),
            "array_pop_back" => Some("arrayPopBack"),
            "array_pop_front" => Some("arrayPopFront"),
            "array_position" => Some("indexOf"),
            "array_prepend" => Some("arrayPushFront"),
            "array_append" => Some("arrayPushBack"),
            "array_remove" => Some("arrayFilter"),
            "array_repeat" => Some("arrayWithConstant"),
            "array_reverse" => Some("arrayReverse"),
            "array_slice" => Some("arraySlice"),
            "array_sort" => Some("arraySort"),
            "array_to_string" => Some("arrayStringConcat"),
            "make_array" => Some("array"),
            // String functions
            "character_length" | "char_length" => Some("length"),
            "concat_ws" => Some("concat"),
            // Other common mappings
            _ => None,
        }
    }
}

impl Default for ClickHouseDialect {
    fn default() -> Self {
        Self::new()
    }
}

impl unparser::dialect::Dialect for ClickHouseDialect {
    fn identifier_quote_style(&self, _identifier: &str) -> Option<char> {
        // ClickHouse uses backticks for quoting identifiers
        Some('`')
    }

    fn scalar_function_to_sql_overrides(
        &self,
        unparser: &Unparser,
        func_name: &str,
        args: &[datafusion::logical_expr::Expr],
    ) -> DataFusionResult<Option<ast::Expr>> {
        // Check if we have a mapping for this function
        if let Some(ch_func_name) = Self::get_function_name_mapping(func_name) {
            // Convert each arg to SQL AST
            let sql_args: Vec<ast::FunctionArg> = args
                .iter()
                .map(|arg| {
                    unparser
                        .expr_to_sql(arg)
                        .map(|expr| ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(expr)))
                })
                .collect::<DataFusionResult<Vec<_>>>()?;

            return Ok(Some(ast::Expr::Function(ast::Function {
                name: ObjectName::from(vec![Ident {
                    value: ch_func_name.to_string(),
                    quote_style: None,
                    span: Span::empty(),
                }]),
                args: ast::FunctionArguments::List(ast::FunctionArgumentList {
                    duplicate_treatment: None,
                    args: sql_args,
                    clauses: vec![],
                }),
                filter: None,
                null_treatment: None,
                over: None,
                within_group: vec![],
                parameters: ast::FunctionArguments::None,
                uses_odbc_syntax: false,
            })));
        }
        Ok(None)
    }
}

pub struct ClickHouseTableFactory {
    pool: Arc<ClickHouseConnectionPool>,
}

impl ClickHouseTableFactory {
    pub fn new(pool: impl Into<Arc<ClickHouseConnectionPool>>) -> Self {
        Self { pool: pool.into() }
    }

    pub async fn table_provider(
        &self,
        table_reference: TableReference,
        args: Option<Vec<(String, Arg)>>,
    ) -> Result<Arc<dyn TableProvider + 'static>, Box<dyn std::error::Error + Send + Sync + 'static>>
    {
        let client: &dyn AsyncDbConnection<Client, ()> = &self.pool.client();
        let schema = client.get_schema(&table_reference).await?;
        let table_provider = Arc::new(ClickHouseTable::new(
            table_reference,
            args,
            self.pool.clone(),
            schema,
            Constraints::default(),
        ));

        #[cfg(feature = "clickhouse-federation")]
        let table_provider = Arc::new(
            table_provider
                .create_federated_table_provider()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?,
        );

        Ok(table_provider)
    }
}

#[derive(Debug, Clone)]
pub enum Arg {
    Unsigned(u64),
    Signed(i64),
    String(String),
}

impl From<String> for Arg {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<u64> for Arg {
    fn from(value: u64) -> Self {
        Self::Unsigned(value)
    }
}

impl From<i64> for Arg {
    fn from(value: i64) -> Self {
        Self::Signed(value)
    }
}

impl From<Arg> for Expr {
    fn from(value: Arg) -> Self {
        Expr::value(match value {
            Arg::Unsigned(x) => Value::Number(x.to_string(), false),
            Arg::Signed(x) => Value::Number(x.to_string(), false),
            Arg::String(x) => Value::SingleQuotedString(x),
        })
    }
}

fn into_table_args(args: Vec<(String, Arg)>) -> Vec<FunctionArg> {
    args.into_iter()
        .map(|(name, value)| FunctionArg::Named {
            name: Ident::new(name),
            arg: FunctionArgExpr::Expr(value.into()),
            operator: FunctionArgOperator::Equals,
        })
        .collect()
}

pub struct ClickHouseTable {
    table_reference: TableReference,
    args: Option<Vec<(String, Arg)>>,
    pool: Arc<ClickHouseConnectionPool>,
    schema: SchemaRef,
    constraints: Constraints,
    dialect: Arc<dyn unparser::dialect::Dialect>,
}

impl std::fmt::Debug for ClickHouseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClickHouseTable")
            .field("table_name", &self.table_reference)
            .field("schema", &self.schema)
            .field("constraints", &self.constraints)
            .finish()
    }
}

impl ClickHouseTable {
    pub fn new(
        table_reference: TableReference,
        args: Option<Vec<(String, Arg)>>,
        pool: Arc<ClickHouseConnectionPool>,
        schema: SchemaRef,
        constraints: Constraints,
    ) -> Self {
        Self {
            table_reference,
            args,
            pool,
            schema,
            constraints,
            dialect: Arc::new(ClickHouseDialect::new()),
        }
    }
}
