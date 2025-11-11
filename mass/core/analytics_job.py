#!/usr/bin/env python3

import sys
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import adapters
from ..adapters.base import DataAdapter
from ..adapters.ydb_adapter import YDBAdapter

# Import analytics modules
from .config_loader import ConfigLoader, ConfigError
from .data_access import DataAccess
from .preprocessing import Preprocessing
from .baseline_calculator import BaselineCalculator
from .event_detector import EventDetector
from .persistence import Persistence
from .visualization import EventVisualizer
from .summary_report import SummaryReportGenerator
from .context_tracker import ContextTracker


def create_adapter(config: Dict[str, Any]) -> DataAdapter:
    """
    Create appropriate data adapter based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataAdapter instance
    """
    data_source = config.get('data_source', {})
    
    # Check for YDB configuration
    if 'ydb' in data_source:
        # YDB adapter
        config_path = data_source.get('ydb', {}).get('config_path')
        return YDBAdapter(
            config_path=config_path,
            enable_statistics=None,
            script_name=None,
            silent=False,
            use_local_config=True
        )
    else:
        # Default to YDB for backward compatibility
        # In the future, can add other adapters here
        return YDBAdapter(
            config_path=None,
            enable_statistics=None,
            script_name=None,
            silent=False,
            use_local_config=True
        )


class AnalyticsJob:
    """Main orchestrator for analytics pipeline"""
    
    def __init__(self, config_path: str, dry_run: bool = False, event_deepness: Optional[str] = None, 
                 data_file: Optional[str] = None):
        """
        Initialize analytics job
        
        Args:
            config_path: Path to YAML configuration file
            dry_run: If True, don't write to data source
            event_deepness: Optional time window for event analysis (e.g., "7d", "30d", "1h", "2w")
                          Only events within this window will be analyzed
            data_file: Optional path to saved data file (if provided, load from file instead of data source)
        """
        self.config_path = config_path
        self.dry_run = dry_run
        self.event_deepness = event_deepness
        self.data_file = data_file
        
        # Load configuration
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.get_config()
        except ConfigError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Override dry_run from config if specified
        if 'output' in self.config and 'dry_run' in self.config['output']:
            self.dry_run = self.config['output']['dry_run'] or self.dry_run
        
        # Create data adapter based on configuration
        self.adapter = create_adapter(self.config)
        
        # Initialize components
        self.data_access = DataAccess(self.adapter, self.config)
        self.preprocessing = Preprocessing(self.config)
        self.baseline_calculator = BaselineCalculator(self.config)
        self.event_detector = EventDetector(self.config)
        self.context_tracker = ContextTracker(self.config, self.preprocessing)
        
        # Initialize persistence only if not dry_run
        if not self.dry_run:
            self.persistence = Persistence(self.adapter, self.config)
        else:
            self.persistence = None
        
        # Runtime tracking
        self.start_time = time.time()
        self.max_runtime_minutes = None
        if 'runtime' in self.config and 'max_runtime_minutes' in self.config['runtime']:
            self.max_runtime_minutes = self.config['runtime']['max_runtime_minutes']
    
    def run(self):
        """Run the analytics pipeline"""
        job_name = self.config.get('job', {}).get('name', 'analytics_job')
        print(f"Starting analytics job: {job_name}")
        
        try:
            # Step 1: Load data
            if self.data_file:
                print(f"Step 1: Loading measurements from file: {self.data_file}...")
                df_all = self.data_access.load_data_from_file(self.data_file)
            else:
                print("Step 1: Loading measurements from data source...")
                # Always load ALL data for baseline calculation (stable baseline on historical data)
                df_all = self.data_access.load_measurements(start_ts=None, end_ts=None)
            
            # Calculate time window for event analysis if event_deepness is specified
            event_start_ts = None
            event_end_ts = None
            if self.event_deepness:
                event_end_ts = datetime.now()
                event_start_ts = self._parse_event_deepness(self.event_deepness, event_end_ts)
                print(f"  Analyzing events in window: {event_start_ts.strftime('%Y-%m-%d %H:%M:%S')} to {event_end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Baseline computed on all available data (for stability)")
            
            if df_all.empty:
                print("Warning: No data loaded from data source")
                return
            
            summary = self.data_access.get_data_summary(df_all)
            print(f"Loaded {summary['total_rows']} rows for baseline calculation")
            print(f"Metrics: {', '.join(summary['metrics'])}")
            print(f"Context combinations: {summary['context_combinations']}")
            
            # Check runtime
            self._check_runtime()
            
            # Step 2: Preprocess and group data (on ALL data for baseline)
            print("\nStep 2: Preprocessing and grouping data...")
            df_cleaned = self.preprocessing.clean_data(df_all, remove_outliers=True)
            grouped_data = self.preprocessing.group_by_context(df_cleaned)
            
            print(f"Grouped into {len(grouped_data)} metric×context combinations")
            
            # Detect context changes using context_tracker
            context_changes = self.context_tracker.detect_context_changes(
                df_cleaned, event_start_ts, event_end_ts
            )
            new_error_contexts = context_changes.get('new', set())
            disappeared_error_contexts = context_changes.get('disappeared', set())
            new_with_rules = context_changes.get('new_with_rules', {})
            disappeared_with_rules = context_changes.get('disappeared_with_rules', {})
            
            if new_error_contexts:
                print(f"  ⚠ Detected {len(new_error_contexts)} new context types (appeared in event window)")
            if disappeared_error_contexts:
                print(f"  ✓ Detected {len(disappeared_error_contexts)} disappeared context types (gone in event window)")
            
            # Step 3: Process each group
            print("\nStep 3: Computing baselines and detecting events...")
            all_thresholds = []
            all_events = []
            # Store visualization data for groups with events
            visualization_data = []
            
            processed = 0
            for group_key, group_df in grouped_data.items():
                # Check runtime
                self._check_runtime()
                
                # Validate group has enough data
                min_points = self.config.get('analytics', {}).get('min_data_points', 3)
                has_enough_data = self.preprocessing.validate_group_data(group_df, min_points=min_points)
                
                # Extract metric name and context
                metric_name = group_key[0]
                context_values = self.preprocessing.extract_context_from_group_key(group_key)
                context_hash = self.preprocessing.compute_context_hash(context_values)
                context_json = self._context_to_json(context_values)
                
                # Check if this is a new or disappeared context
                is_new_context = group_key in new_error_contexts
                is_disappeared_context = group_key in disappeared_error_contexts
                new_rule = new_with_rules.get(group_key)
                disappeared_rule = disappeared_with_rules.get(group_key)
                
                if not has_enough_data:
                    # For new contexts: create event even without baseline if rule exists
                    if is_new_context and new_rule:
                        timestamp_field = self.config.get('timestamp_field')
                        if timestamp_field in group_df.columns:
                            first_time = group_df[timestamp_field].min()
                            last_time = group_df[timestamp_field].max()
                            metric_value_field = self.config.get('metric_fields', [None, None])[1]
                            if metric_value_field and metric_value_field in group_df.columns:
                                value = float(group_df[metric_value_field].sum() if 'count' in metric_name.lower() else group_df[metric_value_field].mean())
                            else:
                                value = float(len(group_df))
                            
                            event_data = {
                                'timestamp': first_time,
                                'metric_name': metric_name,
                                'context_hash': context_hash,
                                'context_json': context_json,
                                'event_type': new_rule.get('event_type', 'degradation_start'),
                                'event_start_time': first_time,
                                'event_end_time': last_time if len(group_df) > 1 else first_time,
                                'severity': new_rule.get('severity', 'high'),
                                'baseline_before': new_rule.get('baseline_before', 0.0),
                                'baseline_after': 0.0,
                                'threshold_before': None,
                                'threshold_after': None,
                                'change_absolute': value,
                                'change_relative': float('inf') if value > 0 else 0.0,
                                'current_value': value,
                            }
                            all_events.append(event_data)
                            print(f"  ⚠ New context detected: {context_json[:80]}... (value: {value})")
                            
                            # Create minimal series for visualization even with insufficient data
                            series_all = self.preprocessing.prepare_time_series(group_df)
                            if not series_all.empty:
                                # Create minimal baseline result for visualization
                                baseline_value = new_rule.get('baseline_before', 0.0)
                                baseline_result = {
                                    'baseline_series': pd.Series([baseline_value], index=[first_time]),
                                    'upper_threshold': None,
                                    'lower_threshold': None,
                                    'baseline_value': baseline_value
                                }
                                
                                # Store visualization data
                                visualization_data.append({
                                    'metric_name': metric_name,
                                    'context_hash': context_hash,
                                    'context_json': context_json,
                                    'series': series_all,
                                    'baseline_result': baseline_result,
                                    'events': [event_data],
                                })
                        continue
                    else:
                        print(f"Skipping group {group_key}: insufficient data (need at least {min_points} points, got {len(group_df)})")
                        continue
                
                # Prepare time series from ALL data (for stable baseline)
                series_all = self.preprocessing.prepare_time_series(group_df)
                
                if series_all.empty:
                    continue
                
                # Compute baseline and thresholds on ALL data
                baseline_result = self.baseline_calculator.compute_baseline_and_thresholds(series_all)
                
                # Filter series for event detection if event_deepness is specified
                series_for_events = series_all
                if self.event_deepness and event_start_ts and event_end_ts:
                    timestamp_field = self.config.get('timestamp_field')
                    if timestamp_field:
                        # Filter series to event window
                        mask = (series_all.index >= event_start_ts) & (series_all.index <= event_end_ts)
                        series_for_events = series_all[mask]
                        if len(series_for_events) == 0:
                            # No data in event window - check if context disappeared
                            if is_disappeared_context and disappeared_rule:
                                # Get last occurrence time before window
                                df_before_window = group_df[group_df[timestamp_field] < event_start_ts]
                                if not df_before_window.empty:
                                    last_time = df_before_window[timestamp_field].max()
                                    event_data = {
                                        'timestamp': last_time,
                                        'metric_name': metric_name,
                                        'context_hash': context_hash,
                                        'context_json': context_json,
                                        'event_type': disappeared_rule.get('event_type', 'improvement_start'),
                                        'event_start_time': last_time,
                                        'event_end_time': event_end_ts,
                                        'severity': disappeared_rule.get('severity', 'medium'),
                                        'baseline_before': baseline_result.get('baseline_value', 0.0),
                                        'baseline_after': disappeared_rule.get('baseline_after', 0.0),
                                        'threshold_before': baseline_result.get('upper_threshold'),
                                        'threshold_after': None,
                                        'change_absolute': 0.0,
                                        'change_relative': 0.0,
                                        'current_value': 0.0,
                                    }
                                    all_events.append(event_data)
                                    print(f"  ✓ Disappeared context detected: {context_json[:80]}...")
                            continue
                
                # Save threshold data
                threshold_data = {
                    'timestamp': datetime.now(),
                    'metric_name': metric_name,
                    'context_hash': context_hash,
                    'context_json': context_json,
                    'baseline_value': baseline_result['baseline_value'],
                    'upper_threshold': baseline_result['upper_threshold'],
                    'lower_threshold': baseline_result['lower_threshold'],
                    'baseline_method': baseline_result['baseline_method'],
                    'window_size': baseline_result['window_size'],
                    'sensitivity': baseline_result['sensitivity'],
                    'adaptive_threshold': baseline_result['adaptive_threshold'],
                }
                all_thresholds.append(threshold_data)
                
                # Detect events only in the filtered window (pass metric_name for direction detection)
                events = self.event_detector.detect_events(series_for_events, baseline_result, metric_name=metric_name)
                
                # For new contexts: if no events detected but it's a new context, create event anyway
                # This ensures we always detect appearance of new contexts
                if is_new_context and new_rule and len(events) == 0:
                    # New context appeared but no event detected by normal logic
                    # Create event to mark appearance of new context
                    if not series_for_events.empty:
                        first_time = series_for_events.index[0]
                        # For aggregated data, sum all values; for raw data, use first value
                        value = float(series_for_events.sum() if 'count' in metric_name.lower() else series_for_events.mean())
                        
                        event_data = {
                            'timestamp': first_time,
                            'metric_name': metric_name,
                            'context_hash': context_hash,
                            'context_json': context_json,
                            'event_type': new_rule.get('event_type', 'degradation_start'),
                            'event_start_time': first_time,
                            'event_end_time': series_for_events.index[-1] if len(series_for_events) > 1 else first_time,
                            'severity': new_rule.get('severity', 'high'),
                            'baseline_before': new_rule.get('baseline_before', 0.0),
                            'baseline_after': float(baseline_result.get('baseline_value') or 0.0),
                            'threshold_before': None,
                            'threshold_after': baseline_result.get('upper_threshold'),
                            'change_absolute': value,
                            'change_relative': float('inf') if value > 0 else 0.0,
                            'current_value': value,
                        }
                        events.append(event_data)
                        print(f"  ⚠ New context detected (with baseline): {context_json[:80]}... (value: {value})")
                
                # Debug logging for event detection
                if self.config.get('output', {}).get('log_to_console', False):
                    baseline_val = baseline_result.get('baseline_value')
                    upper = baseline_result.get('upper_threshold')
                    lower = baseline_result.get('lower_threshold')
                    latest_value = float(series_for_events.iloc[-1]) if not series_for_events.empty else None
                    
                    # Count points outside thresholds (in event window)
                    if upper is not None:
                        above_upper = (series_for_events > upper).sum() if not series_for_events.empty else 0
                    else:
                        above_upper = 0
                    
                    if lower is not None:
                        below_lower = (series_for_events < lower).sum() if not series_for_events.empty else 0
                    else:
                        below_lower = 0
                    
                    if len(events) == 0 and (above_upper > 0 or below_lower > 0):
                        print(f"  Debug: {context_json[:60]}... baseline={baseline_val:.2f}, upper={upper}, lower={lower}, latest={latest_value:.2f}, above_upper={above_upper}, below_lower={below_lower}")
                
                # Add context info to events
                for event in events:
                    event['metric_name'] = metric_name
                    event['context_hash'] = context_hash
                    event['context_json'] = context_json
                
                all_events.extend(events)
                
                # Store visualization data - always store for exploration mode (dry_run)
                # For exploration, we want to see all groups even without events
                # Use series_all for visualization to show full context, but events are only in the window
                if events or self.dry_run:
                    visualization_data.append({
                        'metric_name': metric_name,
                        'context_hash': context_hash,
                        'context_json': context_json,
                        'series': series_all,  # Show full series for context
                        'baseline_result': baseline_result,
                        'events': events,  # Events are only in the filtered window
                        'variant_config': {
                            'baseline_method': baseline_result.get('baseline_method'),
                            'window_size': baseline_result.get('window_size'),
                            'sensitivity': baseline_result.get('sensitivity'),
                            'adaptive_threshold': baseline_result.get('adaptive_threshold', False),
                            'analytics_config': self.analytics_config.copy() if hasattr(self, 'analytics_config') else {}
                        }
                    })
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{len(grouped_data)} groups...")
            
            print(f"\nComputed baselines for {len(all_thresholds)} groups")
            print(f"Detected {len(all_events)} events")
            
            # Step 4: Save results
            if not self.dry_run:
                print("\nStep 4: Saving results to data source...")
                
                # Ensure tables exist
                if self.persistence:
                    self.persistence.ensure_tables_exist()
                
                # Save thresholds
                if all_thresholds:
                    keep_history = self.config.get('thresholds', {}).get('keep_history', True)
                    if keep_history:
                        if self.persistence:
                            self.persistence.save_thresholds(all_thresholds)
                            print(f"Saved {len(all_thresholds)} threshold records")
                    else:
                        print("Threshold history disabled, skipping threshold save")
                
                # Save events
                if all_events:
                    if self.persistence:
                        self.persistence.save_events(all_events)
                        print(f"Saved {len(all_events)} event records")
            else:
                print("\nStep 4: Dry run - skipping save to data source")
                self._save_dry_run_results(all_events, all_thresholds, visualization_data)
            
            # Step 5: Generate visualizations and reports
            # Always generate visualizations in dry_run mode (for exploration)
            # Generate visualizations if we have data to visualize (even without events/baselines)
            if visualization_data:
                print("\nStep 5: Generating visualizations and reports...")
                print(f"  Found {len(visualization_data)} metric×context combinations")
                
                # Group visualization data by context_hash
                from collections import defaultdict
                visualization_by_context = defaultdict(list)
                for viz_data in visualization_data:
                    context_hash = viz_data.get('context_hash', '')
                    visualization_by_context[context_hash].append(viz_data)
                
                print(f"  Grouped into {len(visualization_by_context)} contexts to visualize")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                job_name = self.config.get('job', {}).get('name', 'analytics_job')
                output_dir = self.config.get('output', {}).get('output_dir', 'dry_run_output')
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Enhance visualization_data with all events for each context+metric combination
                # This ensures that visualizations show all events, not just those from the processing window
                for context_hash, metrics_data in visualization_by_context.items():
                    for metric_data in metrics_data:
                        metric_name = metric_data.get('metric_name', '')
                        # Find all events for this context_hash and metric_name
                        matching_events = [
                            event for event in all_events
                            if event.get('context_hash') == context_hash and event.get('metric_name') == metric_name
                        ]
                        # Replace events with all matching events
                        metric_data['events'] = matching_events
                
                # Generate visualizations - pass grouped data
                EventVisualizer.generate_visualizations(visualization_by_context, output_dir, timestamp, job_name)
                
                # Generate summary report (even if no events, for exploration)
                if all_events or (self.dry_run and all_thresholds):
                    SummaryReportGenerator.generate_summary_html(
                        all_events, visualization_data, output_dir, timestamp, job_name
                    )
                    print(f"  ✓ Generated summary report: {output_dir}/{job_name}_summary_{timestamp}.html")
            elif self.dry_run:
                print("\nStep 5: No visualization data available")
                print(f"  visualization_data: {len(visualization_data) if visualization_data else 0} groups")
                print(f"  all_thresholds: {len(all_thresholds)} thresholds")
                print(f"  all_events: {len(all_events)} events")
            
            elapsed = time.time() - self.start_time
            print(f"\n✓ Analytics job completed successfully in {elapsed:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n⚠ Analytics job interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error during analytics job: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _parse_event_deepness(self, deepness_str: str, end_ts: datetime) -> datetime:
        """
        Parse event deepness string to datetime
        
        Args:
            deepness_str: String like "7d", "30d", "1h", "2w"
            end_ts: End timestamp
            
        Returns:
            Start timestamp
        """
        import re
        match = re.match(r'^(\d+)([dhwms])$', deepness_str.lower())
        if not match:
            raise ValueError(f"Invalid event_deepness format: {deepness_str}. Use format like '7d', '30d', '1h', '2w'")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'd':
            delta = timedelta(days=value)
        elif unit == 'h':
            delta = timedelta(hours=value)
        elif unit == 'm':
            delta = timedelta(minutes=value)
        elif unit == 's':
            delta = timedelta(seconds=value)
        elif unit == 'w':
            delta = timedelta(weeks=value)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
        
        return end_ts - delta
    
    def _check_runtime(self):
        """Check if job has exceeded max runtime"""
        if self.max_runtime_minutes is None:
            return
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes >= self.max_runtime_minutes:
            raise RuntimeError(f"Job exceeded max runtime of {self.max_runtime_minutes} minutes")
    
    def _context_to_json(self, context_values: Dict[str, Any]) -> str:
        """Convert context values to JSON string"""
        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_value(v):
            # Check for numpy types
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                return str(v)
            elif isinstance(v, pd.Series):
                return v.tolist()
            # Check for numpy scalar types by class name (fallback)
            elif type(v).__name__ in ['int64', 'int32', 'int16', 'int8', 
                                       'uint64', 'uint32', 'uint16', 'uint8',
                                       'float64', 'float32', 'float16',
                                       'bool_']:
                try:
                    return v.item()
                except (AttributeError, ValueError):
                    return int(v) if isinstance(v, (np.integer,)) else float(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            else:
                return v
        
        converted_values = {k: convert_value(v) for k, v in context_values.items()}
        return json.dumps(converted_values, sort_keys=True, ensure_ascii=False, default=str)
    
    def _save_dry_run_results(self, events: List[Dict[str, Any]], 
                             thresholds: List[Dict[str, Any]],
                             visualization_data: List[Dict[str, Any]] = None):
        """
        Save dry run results to files
        
        Args:
            events: List of event dictionaries
            thresholds: List of threshold dictionaries
            visualization_data: List of dicts with series, baseline_result, events for visualization
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_name = self.config.get('job', {}).get('name', 'analytics_job')
        output_dir = self.config.get('output', {}).get('output_dir', 'dry_run_output')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save events to JSON
        if events:
            events_file = os.path.join(output_dir, f"{job_name}_events_{timestamp}.json")
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, default=str, ensure_ascii=False)
            print(f"  ✓ Saved {len(events)} events to {events_file}")
        
        # Save events to CSV
        if events:
            events_csv = os.path.join(output_dir, f"{job_name}_events_{timestamp}.csv")
            events_df = pd.DataFrame(events)
            # Convert timestamp columns to strings for CSV
            for col in events_df.columns:
                if 'time' in col.lower() or col == 'timestamp':
                    if pd.api.types.is_datetime64_any_dtype(events_df[col]):
                        events_df[col] = events_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            events_df.to_csv(events_csv, index=False, encoding='utf-8')
            print(f"  ✓ Saved {len(events)} events to {events_csv}")
        
        # Save thresholds to CSV
        if thresholds:
            thresholds_file = os.path.join(output_dir, f"{job_name}_thresholds_{timestamp}.csv")
            thresholds_df = pd.DataFrame(thresholds)
            # Convert timestamp to string for CSV
            if 'timestamp' in thresholds_df.columns:
                if pd.api.types.is_datetime64_any_dtype(thresholds_df['timestamp']):
                    thresholds_df['timestamp'] = thresholds_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            thresholds_df.to_csv(thresholds_file, index=False, encoding='utf-8')
            print(f"  ✓ Saved {len(thresholds)} thresholds to {thresholds_file}")
        
        # NOTE: Visualizations and reports are generated in Step 5, not here
        # to avoid duplication. This method only saves raw data (events, thresholds).


def main():
    """Main entry point for analytics job"""
    parser = argparse.ArgumentParser(description='Run analytics job')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Run without saving to data source')
    parser.add_argument('--event-deepness', type=str, help='Time window for event analysis (e.g., "7d", "30d", "1h", "2w")')
    
    args = parser.parse_args()
    
    job = AnalyticsJob(args.config, dry_run=args.dry_run, event_deepness=args.event_deepness)
    job.run()


if __name__ == '__main__':
    main()

