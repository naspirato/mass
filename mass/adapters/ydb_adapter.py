"""
YDB adapter using ydb_wrapper
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from .base import DataAdapter
import sys
import os

# Add utils to path for ydb_wrapper import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from ydb_wrapper import YDBWrapper


class YDBAdapter(DataAdapter):
    """Adapter for YDB data source using ydb_wrapper"""
    
    def __init__(self, config_path: str = None, enable_statistics: bool = None, 
                 script_name: str = None, silent: bool = False, use_local_config: bool = True):
        """
        Initialize YDB adapter
        
        Args:
            config_path: Path to YDB config file
            enable_statistics: Enable statistics logging
            script_name: Script name for logging
            silent: Silent mode (logs to stderr)
            use_local_config: Use local config file only
        """
        self.ydb_wrapper = YDBWrapper(
            config_path=config_path,
            enable_statistics=enable_statistics,
            script_name=script_name,
            silent=silent,
            use_local_config=use_local_config
        )
    
    def load_data(self, query: str, query_name: str = None, **kwargs) -> pd.DataFrame:
        """
        Load data from YDB using scan query
        
        Args:
            query: SQL query string
            query_name: Optional query name for logging
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with loaded data
        """
        results = self.ydb_wrapper.execute_scan_query(query, query_name=query_name)
        
        if not results:
            return pd.DataFrame()
        
        # Convert YDB row objects to dictionaries
        dict_results = []
        for row in results:
            row_dict = dict(row)
            # Convert bytes values to strings
            for key, value in row_dict.items():
                if isinstance(value, bytes):
                    row_dict[key] = value.decode('utf-8')
            dict_results.append(row_dict)
        
        return pd.DataFrame(dict_results)
    
    def save_data(self, table_path: str, data: List[Dict[str, Any]], 
                  column_types=None, batch_size: int = 1000, query_name: str = None, **kwargs) -> bool:
        """
        Save data to YDB using bulk upsert
        
        Args:
            table_path: Relative path to table
            data: List of dictionaries to save
            column_types: YDB column types (BulkUpsertColumns)
            batch_size: Batch size for bulk upsert
            query_name: Optional query name for logging
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        if not data:
            return True
        
        if column_types is None:
            raise ValueError("column_types is required for YDB bulk upsert")
        
        if len(data) > batch_size:
            # Use batched upsert
            self.ydb_wrapper.bulk_upsert_batches(
                table_path, data, column_types, batch_size=batch_size, query_name=query_name
            )
        else:
            # Use single bulk upsert
            self.ydb_wrapper.bulk_upsert(table_path, data, column_types)
        
        return True
    
    def execute_query(self, query: str, query_name: str = None, **kwargs) -> Any:
        """
        Execute a query (DML, DDL, etc.)
        
        Args:
            query: SQL query string
            query_name: Optional query name for logging
            **kwargs: Additional parameters
            
        Returns:
            Query result
        """
        # For DML queries
        if any(keyword in query.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE']):
            parameters = kwargs.get('parameters', {})
            return self.ydb_wrapper.execute_dml(query, parameters=parameters, query_name=query_name)
        else:
            # For other queries, use scan query
            return self.ydb_wrapper.execute_scan_query(query, query_name=query_name)
    
    def create_table(self, table_path: str, schema: str, **kwargs) -> bool:
        """
        Create a table in YDB
        
        Args:
            table_path: Relative path to table
            schema: SQL CREATE TABLE statement
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        self.ydb_wrapper.create_table(table_path, schema)
        return True
    
    def get_driver(self):
        """Get YDB driver context manager (for advanced usage)"""
        return self.ydb_wrapper.get_driver()

