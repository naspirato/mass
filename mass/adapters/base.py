"""
Base class for data source adapters
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class DataAdapter(ABC):
    """Base class for data source adapters"""
    
    @abstractmethod
    def load_data(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Load data from data source
        
        Args:
            query: Query string (SQL, file path, etc. depending on adapter)
            **kwargs: Additional parameters for the query
            
        Returns:
            DataFrame with loaded data
        """
        pass
    
    @abstractmethod
    def save_data(self, table_path: str, data: List[Dict[str, Any]], **kwargs) -> bool:
        """
        Save data to data source
        
        Args:
            table_path: Path/name of the table/collection
            data: List of dictionaries to save
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, **kwargs) -> Any:
        """
        Execute a query (for operations that don't return data)
        
        Args:
            query: Query string
            **kwargs: Additional parameters
            
        Returns:
            Query result (depends on adapter)
        """
        pass
    
    @abstractmethod
    def create_table(self, table_path: str, schema: Any, **kwargs) -> bool:
        """
        Create a table/collection if it doesn't exist
        
        Args:
            table_path: Path/name of the table/collection
            schema: Schema definition (depends on adapter)
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        pass

