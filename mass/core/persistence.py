#!/usr/bin/env python3

import ydb
import hashlib
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..adapters.base import DataAdapter
from ..adapters.ydb_adapter import YDBAdapter


class Persistence:
    """Persistence layer for saving analytics results to data sources"""
    
    def __init__(self, adapter: DataAdapter, config: Dict[str, Any]):
        """
        Initialize persistence layer
        
        Args:
            adapter: DataAdapter instance
            config: Configuration dictionary
        """
        self.adapter = adapter
        self.config = config
        
        # Get session info from adapter if it's YDBAdapter
        if isinstance(adapter, YDBAdapter):
            ydb_wrapper = adapter.ydb_wrapper
            self.session_id = ydb_wrapper._session_id
            self.script_name = ydb_wrapper._script_name
            self.cluster_version = ydb_wrapper._cluster_version or "unknown"
            
            # Get table paths from config or use defaults
            try:
                self.thresholds_table = ydb_wrapper.get_table_path("analytics_thresholds", "main")
            except KeyError:
                self.thresholds_table = "analytics/thresholds"
            
            try:
                self.events_table = ydb_wrapper.get_table_path("analytics_events", "main")
            except KeyError:
                self.events_table = "analytics/events"
        else:
            # For non-YDB adapters, use defaults
            self.session_id = "default_session"
            self.script_name = "analytics_job"
            self.cluster_version = "unknown"
            self.thresholds_table = "analytics/thresholds"
            self.events_table = "analytics/events"
    
    def ensure_tables_exist(self):
        """Create tables if they don't exist"""
        if isinstance(self.adapter, YDBAdapter):
            self._create_thresholds_table()
            self._create_events_table()
        # For other adapters, table creation might be handled differently
        # This can be extended in the future
    
    def _create_thresholds_table(self):
        """Create analytics_thresholds table if it doesn't exist (YDB-specific)"""
        if not isinstance(self.adapter, YDBAdapter):
            return
            
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.thresholds_table}` (
            `timestamp` Timestamp NOT NULL,
            `session_id` Utf8 NOT NULL,
            `metric_name` Utf8 NOT NULL,
            `context_hash` Utf8 NOT NULL,
            `baseline_value` Double,
            `upper_threshold` Double,
            `lower_threshold` Double,
            `baseline_method` Utf8 NOT NULL,
            `window_size` Uint64,
            `sensitivity` Double,
            `adaptive_threshold` Bool,
            `cluster_version` Utf8,
            `script_name` Utf8 NOT NULL,
            `context_json` Utf8,
            PRIMARY KEY (`timestamp`, `session_id`, `metric_name`, `context_hash`)
        )
        PARTITION BY HASH(`session_id`, `metric_name`)
        WITH (STORE = COLUMN)
        """
        
        self.adapter.create_table(self.thresholds_table, create_sql)
    
    def _create_events_table(self):
        """Create analytics_events table if it doesn't exist (YDB-specific)"""
        if not isinstance(self.adapter, YDBAdapter):
            return
            
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.events_table}` (
            `timestamp` Timestamp NOT NULL,
            `session_id` Utf8 NOT NULL,
            `event_type` Utf8 NOT NULL,
            `metric_name` Utf8 NOT NULL,
            `context_hash` Utf8 NOT NULL,
            `event_start_time` Timestamp NOT NULL,
            `event_end_time` Timestamp,
            `severity` Utf8,
            `baseline_before` Double,
            `baseline_after` Double,
            `threshold_before` Double,
            `threshold_after` Double,
            `change_absolute` Double,
            `change_relative` Double,
            `current_value` Double,
            `cluster_version` Utf8,
            `script_name` Utf8 NOT NULL,
            `context_json` Utf8,
            PRIMARY KEY (`timestamp`, `session_id`, `event_type`, `metric_name`, `context_hash`)
        )
        PARTITION BY HASH(`session_id`, `metric_name`)
        WITH (STORE = COLUMN)
        """
        
        self.adapter.create_table(self.events_table, create_sql)
    
    def save_thresholds(self, thresholds_data: List[Dict[str, Any]]):
        """
        Save thresholds to data source
        
        Args:
            thresholds_data: List of threshold dictionaries
        """
        if not thresholds_data:
            return
        
        # Prepare rows for bulk upsert
        rows = []
        for threshold in thresholds_data:
            row = {
                'timestamp': self._to_ydb_timestamp(threshold['timestamp']),
                'session_id': self.session_id,
                'metric_name': str(threshold['metric_name']),
                'context_hash': str(threshold['context_hash']),
                'baseline_value': threshold.get('baseline_value'),
                'upper_threshold': threshold.get('upper_threshold'),
                'lower_threshold': threshold.get('lower_threshold'),
                'baseline_method': str(threshold['baseline_method']),
                'window_size': threshold.get('window_size'),
                'sensitivity': threshold.get('sensitivity'),
                'adaptive_threshold': threshold.get('adaptive_threshold', False),
                'cluster_version': self.cluster_version,
                'script_name': self.script_name,
                'context_json': threshold.get('context_json'),
            }
            rows.append(row)
        
        # For YDB, define column types
        if isinstance(self.adapter, YDBAdapter):
            column_types = (
                ydb.BulkUpsertColumns()
                .add_column("timestamp", ydb.OptionalType(ydb.PrimitiveType.Timestamp))
                .add_column("session_id", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("metric_name", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("context_hash", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("baseline_value", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("upper_threshold", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("lower_threshold", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("baseline_method", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("window_size", ydb.OptionalType(ydb.PrimitiveType.Uint64))
                .add_column("sensitivity", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("adaptive_threshold", ydb.OptionalType(ydb.PrimitiveType.Bool))
                .add_column("cluster_version", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("script_name", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("context_json", ydb.OptionalType(ydb.PrimitiveType.Utf8))
            )
            
            # Use bulk_upsert_batches for efficient writing
            query_name = f"save_thresholds_{self.config.get('job', {}).get('name', 'analytics')}"
            self.adapter.save_data(
                self.thresholds_table,
                rows,
                column_types=column_types,
                batch_size=1000,
                query_name=query_name
            )
        else:
            # For other adapters, save_data should handle the conversion
            self.adapter.save_data(self.thresholds_table, rows)
    
    def save_events(self, events_data: List[Dict[str, Any]]):
        """
        Save events to data source
        
        Args:
            events_data: List of event dictionaries
        """
        if not events_data:
            return
        
        # Prepare rows for bulk upsert
        rows = []
        for event in events_data:
            row = {
                'timestamp': self._to_ydb_timestamp(event.get('timestamp', event.get('event_start_time'))),
                'session_id': self.session_id,
                'event_type': str(event['event_type']),
                'metric_name': str(event['metric_name']),
                'context_hash': str(event['context_hash']),
                'event_start_time': self._to_ydb_timestamp(event['event_start_time']),
                'event_end_time': self._to_ydb_timestamp(event.get('event_end_time')),
                'severity': event.get('severity'),
                'baseline_before': event.get('baseline_before'),
                'baseline_after': event.get('baseline_after'),
                'threshold_before': event.get('threshold_before'),
                'threshold_after': event.get('threshold_after'),
                'change_absolute': event.get('change_absolute'),
                'change_relative': event.get('change_relative'),
                'current_value': event.get('current_value'),
                'cluster_version': self.cluster_version,
                'script_name': self.script_name,
                'context_json': event.get('context_json'),
            }
            rows.append(row)
        
        # For YDB, define column types
        if isinstance(self.adapter, YDBAdapter):
            column_types = (
                ydb.BulkUpsertColumns()
                .add_column("timestamp", ydb.OptionalType(ydb.PrimitiveType.Timestamp))
                .add_column("session_id", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("event_type", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("metric_name", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("context_hash", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("event_start_time", ydb.OptionalType(ydb.PrimitiveType.Timestamp))
                .add_column("event_end_time", ydb.OptionalType(ydb.PrimitiveType.Timestamp))
                .add_column("severity", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("baseline_before", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("baseline_after", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("threshold_before", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("threshold_after", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("change_absolute", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("change_relative", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("current_value", ydb.OptionalType(ydb.PrimitiveType.Double))
                .add_column("cluster_version", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("script_name", ydb.OptionalType(ydb.PrimitiveType.Utf8))
                .add_column("context_json", ydb.OptionalType(ydb.PrimitiveType.Utf8))
            )
            
            # Use bulk_upsert_batches for efficient writing
            query_name = f"save_events_{self.config.get('job', {}).get('name', 'analytics')}"
            self.adapter.save_data(
                self.events_table,
                rows,
                column_types=column_types,
                batch_size=1000,
                query_name=query_name
            )
        else:
            # For other adapters, save_data should handle the conversion
            self.adapter.save_data(self.events_table, rows)
    
    def _to_ydb_timestamp(self, value: Any) -> Optional[int]:
        """
        Convert timestamp to YDB timestamp (microseconds since epoch)
        
        Args:
            value: Timestamp value (datetime, pd.Timestamp, or string)
            
        Returns:
            Timestamp in microseconds or None
        """
        if value is None:
            return None
        
        try:
            if isinstance(value, pd.Timestamp):
                dt = value.to_pydatetime()
            elif isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                dt = pd.Timestamp(value).to_pydatetime()
            else:
                return None
            
            # Convert to microseconds since epoch
            epoch = datetime(1970, 1, 1)
            delta = dt - epoch
            return int(delta.total_seconds() * 1_000_000)
        except Exception:
            return None

