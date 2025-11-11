#!/usr/bin/env python3
"""
Exploration module for running analytics with multiple configuration variants
Allows comparing different analytics settings on the same data
"""

import os
import sys
import yaml
import copy
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import concurrent.futures

from .config_loader import ConfigLoader
from .analytics_job import AnalyticsJob


class ExplorationRunner:
    """Run analytics with multiple configuration variants"""
    
    def __init__(self, base_config_path: str, data_file: Optional[str] = None):
        """
        Initialize exploration runner
        
        Args:
            base_config_path: Path to base configuration file
            data_file: Optional path to saved data file (if provided, use saved data instead of loading from source)
        """
        self.base_config_path = base_config_path
        self.base_config = None
        self.data_file = data_file
        self.load_base_config()
    
    def load_base_config(self):
        """Load base configuration"""
        try:
            self.base_config_loader = ConfigLoader(self.base_config_path)
            self.base_config = self.base_config_loader.get_config()
        except Exception as e:
            raise ValueError(f"Failed to load base config: {e}")
    
    def generate_variants(self, variant_settings: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Generate configuration variants based on settings
        
        Args:
            variant_settings: Dictionary with boolean flags for which parameters to vary
                - vary_baseline_method: Vary baseline_method
                - vary_window_size: Vary window_size
                - vary_sensitivity: Vary sensitivity
                - vary_adaptive_threshold: Vary adaptive_threshold
                - vary_min_absolute_change: Vary min_absolute_change
                - vary_min_relative_change: Vary min_relative_change
                - vary_hysteresis_points: Vary hysteresis_points
                - vary_min_data_points: Vary min_data_points
                - vary_min_event_duration: Vary min_event_duration_minutes
                - vary_detect_types: Vary event types to detect
                - vary_track_new_contexts: Vary track_new_contexts
                - vary_track_disappeared_contexts: Vary track_disappeared_contexts
        
        Returns:
            List of variant configuration dictionaries
        """
        variants = []
        
        # Define parameter values to try
        baseline_methods = ['rolling_mean', 'median', 'zscore'] if variant_settings.get('vary_baseline_method', False) else [None]
        window_sizes = [7, 14, 30] if variant_settings.get('vary_window_size', False) else [None]
        sensitivities = [1.5, 2.0, 2.5] if variant_settings.get('vary_sensitivity', False) else [None]
        adaptive_thresholds = [True, False] if variant_settings.get('vary_adaptive_threshold', False) else [None]
        
        # New parameter values
        min_absolute_changes = [0, 5, 10] if variant_settings.get('vary_min_absolute_change', False) else [None]
        min_relative_changes = [0.0, 0.05, 0.1] if variant_settings.get('vary_min_relative_change', False) else [None]
        hysteresis_pointss = [1, 2, 3] if variant_settings.get('vary_hysteresis_points', False) else [None]
        min_data_pointss = [3, 5, 10] if variant_settings.get('vary_min_data_points', False) else [None]
        min_event_durations = [15, 30, 60] if variant_settings.get('vary_min_event_duration', False) else [None]
        
        # Event types to detect
        detect_types_variants = [
            ['degradation_start', 'improvement_start'],
            ['degradation_start'],
            ['improvement_start']
        ] if variant_settings.get('vary_detect_types', False) else [None]
        
        # Context tracking
        track_new_contexts_variants = [True, False] if variant_settings.get('vary_track_new_contexts', False) else [None]
        track_disappeared_contexts_variants = [True, False] if variant_settings.get('vary_track_disappeared_contexts', False) else [None]
        
        # Get base values from config
        base_analytics = self.base_config.get('analytics', {})
        base_events = self.base_config.get('events', {})
        base_context_tracking = self.base_config.get('context_tracking', {})
        
        base_baseline_method = base_analytics.get('baseline_method', 'rolling_mean')
        base_window_size = base_analytics.get('window_size', 7)
        base_sensitivity = base_analytics.get('sensitivity', 2.0)
        base_adaptive_threshold = base_analytics.get('adaptive_threshold', True)
        base_min_absolute_change = base_analytics.get('min_absolute_change', 0)
        base_min_relative_change = base_analytics.get('min_relative_change', 0.0)
        base_hysteresis_points = base_analytics.get('hysteresis_points', 2)
        base_min_data_points = base_analytics.get('min_data_points', 3)
        base_min_event_duration = base_events.get('min_event_duration_minutes', 30)
        base_detect = base_events.get('detect', ['degradation_start', 'improvement_start'])
        base_track_new = base_context_tracking.get('track_new_contexts', False)
        base_track_disappeared = base_context_tracking.get('track_disappeared_contexts', False)
        
        # Generate all combinations
        variant_id = 0
        for baseline_method in baseline_methods:
            for window_size in window_sizes:
                for sensitivity in sensitivities:
                    for adaptive_threshold in adaptive_thresholds:
                        for min_absolute_change in min_absolute_changes:
                            for min_relative_change in min_relative_changes:
                                for hysteresis_points in hysteresis_pointss:
                                    for min_data_points in min_data_pointss:
                                        for min_event_duration in min_event_durations:
                                            for detect_types in detect_types_variants:
                                                for track_new in track_new_contexts_variants:
                                                    for track_disappeared in track_disappeared_contexts_variants:
                                                        # Create variant config
                                                        variant_config = copy.deepcopy(self.base_config)
                                                        
                                                        # Apply variant values (use base if None)
                                                        variant_config['analytics']['baseline_method'] = baseline_method if baseline_method is not None else base_baseline_method
                                                        variant_config['analytics']['window_size'] = window_size if window_size is not None else base_window_size
                                                        variant_config['analytics']['sensitivity'] = sensitivity if sensitivity is not None else base_sensitivity
                                                        variant_config['analytics']['adaptive_threshold'] = adaptive_threshold if adaptive_threshold is not None else base_adaptive_threshold
                                                        
                                                        # New analytics parameters
                                                        if min_absolute_change is not None:
                                                            variant_config['analytics']['min_absolute_change'] = min_absolute_change
                                                        elif 'min_absolute_change' not in variant_config['analytics']:
                                                            variant_config['analytics']['min_absolute_change'] = base_min_absolute_change
                                                        
                                                        if min_relative_change is not None:
                                                            variant_config['analytics']['min_relative_change'] = min_relative_change
                                                        elif 'min_relative_change' not in variant_config['analytics']:
                                                            variant_config['analytics']['min_relative_change'] = base_min_relative_change
                                                        
                                                        if hysteresis_points is not None:
                                                            variant_config['analytics']['hysteresis_points'] = hysteresis_points
                                                        elif 'hysteresis_points' not in variant_config['analytics']:
                                                            variant_config['analytics']['hysteresis_points'] = base_hysteresis_points
                                                        
                                                        if min_data_points is not None:
                                                            variant_config['analytics']['min_data_points'] = min_data_points
                                                        elif 'min_data_points' not in variant_config['analytics']:
                                                            variant_config['analytics']['min_data_points'] = base_min_data_points
                                                        
                                                        # Events parameters
                                                        if min_event_duration is not None:
                                                            variant_config['events']['min_event_duration_minutes'] = min_event_duration
                                                        elif 'min_event_duration_minutes' not in variant_config['events']:
                                                            variant_config['events']['min_event_duration_minutes'] = base_min_event_duration
                                                        
                                                        if detect_types is not None:
                                                            variant_config['events']['detect'] = detect_types
                                                        
                                                        # Context tracking
                                                        if track_new is not None or track_disappeared is not None:
                                                            if 'context_tracking' not in variant_config:
                                                                variant_config['context_tracking'] = {}
                                                            if track_new is not None:
                                                                variant_config['context_tracking']['track_new_contexts'] = track_new
                                                            if track_disappeared is not None:
                                                                variant_config['context_tracking']['track_disappeared_contexts'] = track_disappeared
                                                        
                                                        # Update job name
                                                        job_name = variant_config.get('job', {}).get('name', 'analytics_job')
                                                        variant_config['job']['name'] = f"{job_name}_explore_{variant_id}"
                                                        
                                                        # Store variant metadata
                                                        variant_metadata = {
                                                            'id': variant_id,
                                                            'baseline_method': variant_config['analytics']['baseline_method'],
                                                            'window_size': variant_config['analytics']['window_size'],
                                                            'sensitivity': variant_config['analytics']['sensitivity'],
                                                            'adaptive_threshold': variant_config['analytics']['adaptive_threshold'],
                                                            'min_absolute_change': variant_config['analytics'].get('min_absolute_change', base_min_absolute_change),
                                                            'min_relative_change': variant_config['analytics'].get('min_relative_change', base_min_relative_change),
                                                            'hysteresis_points': variant_config['analytics'].get('hysteresis_points', base_hysteresis_points),
                                                            'min_data_points': variant_config['analytics'].get('min_data_points', base_min_data_points),
                                                            'min_event_duration_minutes': variant_config['events'].get('min_event_duration_minutes', base_min_event_duration),
                                                            'detect': variant_config['events'].get('detect', base_detect),
                                                            'track_new_contexts': variant_config.get('context_tracking', {}).get('track_new_contexts', base_track_new),
                                                            'track_disappeared_contexts': variant_config.get('context_tracking', {}).get('track_disappeared_contexts', base_track_disappeared),
                                                        }
                                                        
                                                        variants.append({
                                                            'config': variant_config,
                                                            'metadata': variant_metadata,
                                                            'id': variant_id
                                                        })
                                                        
                                                        variant_id += 1
        
        # If no variants were generated (all None), create at least one with base config
        if not variants:
            variant_config = copy.deepcopy(self.base_config)
            job_name = variant_config.get('job', {}).get('name', 'analytics_job')
            variant_config['job']['name'] = f"{job_name}_explore_0"
            variants.append({
                'config': variant_config,
                'metadata': {
                    'id': 0,
                    'baseline_method': base_baseline_method,
                    'window_size': base_window_size,
                    'sensitivity': base_sensitivity,
                    'adaptive_threshold': base_adaptive_threshold,
                    'min_absolute_change': base_min_absolute_change,
                    'min_relative_change': base_min_relative_change,
                    'hysteresis_points': base_hysteresis_points,
                    'min_data_points': base_min_data_points,
                    'min_event_duration_minutes': base_min_event_duration,
                    'detect': base_detect,
                    'track_new_contexts': base_track_new,
                    'track_disappeared_contexts': base_track_disappeared,
                },
                'id': 0
            })
        
        return variants
    
    def run_variant(self, variant: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """
        Run analytics for a single variant
        
        Args:
            variant: Variant dictionary with 'config' and 'metadata'
            dry_run: Whether to run in dry-run mode
        
        Returns:
            Dictionary with results: events_count, baselines_count, report_path, variant metadata
        """
        variant_id = variant['id']
        variant_config = variant['config']
        variant_metadata = variant['metadata']
        
        # Create temporary config file for this variant
        # Find project root by looking for configs directory
        base_path = Path(self.base_config_path).resolve()
        project_root = base_path.parent
        # If base_config_path is in configs/, go up one level
        if project_root.name == 'configs':
            project_root = project_root.parent
        # Ensure configs directory exists
        temp_config_dir = project_root / 'configs'
        temp_config_dir.mkdir(exist_ok=True)
        
        temp_config_path = temp_config_dir / f"temp_explore_{variant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        try:
            # Write variant config to temporary file
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(variant_config, f, default_flow_style=False, allow_unicode=True)
            
            # Run analytics job with saved data file if available
            job = AnalyticsJob(str(temp_config_path), dry_run=dry_run, data_file=self.data_file)
            job.run()
            
            # Store baseline_result for recommendations (we'll get it from thresholds CSV)
            baseline_result_for_recs = None
            
            # Find generated reports
            output_dir = variant_config.get('output', {}).get('output_dir', 'dry_run_output')
            # Try multiple possible locations for output directory
            possible_output_paths = [
                project_root / output_dir,  # Standard location
                project_root / 'mass' / 'ui' / output_dir,  # UI location
                Path.cwd() / output_dir,  # Current working directory
                Path.cwd() / 'mass' / 'ui' / output_dir,  # UI from cwd
                base_path.parent / output_dir,  # Relative to config
                base_path.parent / 'mass' / 'ui' / output_dir,  # UI relative to config
            ]
            output_path = None
            for path in possible_output_paths:
                if path.exists() and path.is_dir():
                    output_path = path
                    break
            if output_path is None:
                # Use standard location even if it doesn't exist yet
                output_path = project_root / output_dir
            
            # Count events and baselines from JSON files
            events_count = 0
            positive_events_count = 0
            negative_events_count = 0
            baselines_count = 0
            report_path = None
            visualization_files = []
            events_data = []
            
            # Find the most recent report for this job - search in all possible paths
            job_name = variant_config['job']['name']
            report_files = []
            all_searched_paths = [output_path] + possible_output_paths
            
            for search_path in all_searched_paths:
                if not search_path.exists() or not search_path.is_dir():
                    continue
                found_reports = list(search_path.glob(f"{job_name}_summary_*.html"))
                if found_reports:
                    report_files.extend(found_reports)
                    break  # Found reports, use this path
            
            if report_files:
                # Get the most recent one
                report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                report_path = f"/api/report/{report_files[0].name}"
            
            # Search for event and threshold files in all possible paths
            event_files = []
            threshold_files = []
            
            for search_path in all_searched_paths:
                if not search_path.exists() or not search_path.is_dir():
                    continue
                
                # Find event files
                found_events = list(search_path.glob(f"{job_name}_events_*.json"))
                if found_events:
                    event_files.extend(found_events)
                
                # Find threshold files
                found_thresholds = list(search_path.glob(f"{job_name}_thresholds_*.csv"))
                if found_thresholds:
                    threshold_files.extend(found_thresholds)
            
            if event_files:
                event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                with open(event_files[0], 'r', encoding='utf-8') as f:
                    events_data = json.load(f)
                    events_count = len(events_data) if isinstance(events_data, list) else 0
                    
                    # Count positive (improvement) and negative (degradation) events
                    positive_events_count = 0
                    negative_events_count = 0
                    # Count events by metric
                    events_by_metric = {}
                    # Count events by context_hash + metric_name combination
                    events_by_context_metric = {}
                    if isinstance(events_data, list):
                        for event in events_data:
                            event_type = event.get('event_type', '').lower()
                            metric_name = event.get('metric_name', 'unknown')
                            context_hash = event.get('context_hash', '')
                            
                            # Count overall
                            if 'improvement' in event_type:
                                positive_events_count += 1
                            elif 'degradation' in event_type:
                                negative_events_count += 1
                            
                            # Count by metric (for backward compatibility)
                            if metric_name not in events_by_metric:
                                events_by_metric[metric_name] = {
                                    'total': 0,
                                    'positive': 0,
                                    'negative': 0
                                }
                            events_by_metric[metric_name]['total'] += 1
                            if 'improvement' in event_type:
                                events_by_metric[metric_name]['positive'] += 1
                            elif 'degradation' in event_type:
                                events_by_metric[metric_name]['negative'] += 1
                            
                            # Count by context_hash + metric_name (for accurate per-graph counts)
                            context_metric_key = f"{context_hash}_{metric_name}"
                            if context_metric_key not in events_by_context_metric:
                                events_by_context_metric[context_metric_key] = {
                                    'total': 0,
                                    'positive': 0,
                                    'negative': 0
                                }
                            events_by_context_metric[context_metric_key]['total'] += 1
                            if 'improvement' in event_type:
                                events_by_context_metric[context_metric_key]['positive'] += 1
                            elif 'degradation' in event_type:
                                events_by_context_metric[context_metric_key]['negative'] += 1
            
            if threshold_files:
                threshold_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                try:
                    import pandas as pd
                    df = pd.read_csv(threshold_files[0])
                    baselines_count = len(df)
                    # Get first baseline for recommendations
                    if len(df) > 0:
                        baseline_result_for_recs = {
                            'baseline_value': df.iloc[0].get('baseline_value'),
                            'upper_threshold': df.iloc[0].get('upper_threshold'),
                            'lower_threshold': df.iloc[0].get('lower_threshold'),
                        }
                except Exception:
                    baselines_count = 0
            
            # Search for visualization files if any output path exists
            if any(p.exists() and p.is_dir() for p in all_searched_paths):
                
                # Find visualization files for this job
                # Try searching in all possible output paths, not just the first one found
                viz_files = []
                
                for search_path in all_searched_paths:
                    if not search_path.exists() or not search_path.is_dir():
                        continue
                    
                    # Try exact job_name pattern
                    viz_pattern = f"{job_name}_event_*.html"
                    found_files = list(search_path.glob(viz_pattern))
                    if found_files:
                        viz_files.extend(found_files)
                        break  # Found files, use this path
                    
                    # Try alternative patterns if job_name doesn't have explore suffix
                    if '_explore_' not in job_name:
                        for explore_suffix in ['_explore_0', '_explore_1', '_explore_2']:
                            alt_pattern = f"{job_name}{explore_suffix}_event_*.html"
                            alt_files = list(search_path.glob(alt_pattern))
                            if alt_files:
                                viz_files.extend(alt_files)
                                break
                        if viz_files:
                            break
                
                # If still no files, try fallback search in all paths
                if not viz_files:
                    for search_path in all_searched_paths:
                        if not search_path.exists() or not search_path.is_dir():
                            continue
                        
                        # Try to find files by timestamp from summary report
                        all_event_files = list(search_path.glob("*_event_*.html"))
                        if all_event_files and report_files:
                            # Get timestamp from summary report filename
                            report_filename = report_files[0].stem
                            parts = report_filename.split('_')
                            if len(parts) >= 2:
                                report_timestamp = '_'.join(parts[-2:])
                            else:
                                report_timestamp = parts[-1]
                            
                            # Find files with matching timestamp and job prefix
                            job_prefix = job_name.rsplit('_explore_', 1)[0] if '_explore_' in job_name else job_name
                            matching_files = [
                                f for f in all_event_files 
                                if report_timestamp in f.name and job_prefix in f.name
                            ]
                            if matching_files:
                                viz_files = matching_files
                                break
                        elif all_event_files:
                            # Filter by job prefix
                            job_prefix = job_name.rsplit('_explore_', 1)[0] if '_explore_' in job_name else job_name
                            prefix_files = [f for f in all_event_files if job_prefix in f.name]
                            if prefix_files:
                                viz_files = prefix_files
                                break
                
                if viz_files:
                    # Remove duplicates and sort
                    viz_files = list(set(viz_files))
                    viz_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    
                    # Filter by timestamp from summary report if available (to get only files from this run)
                    if report_files:
                        report_filename = report_files[0].stem
                        parts = report_filename.split('_')
                        if len(parts) >= 2:
                            report_timestamp = '_'.join(parts[-2:])
                            # Only include files with matching timestamp
                            viz_files = [f for f in viz_files if report_timestamp in f.name]
                    
                    # Create list of visualization files with metric info
                    visualization_files = []
                    for f in viz_files:
                        # Extract context hash and metric name from filename: {job}_event_{hash}_{metric}_{timestamp}.html
                        filename = f.name
                        parts = filename.replace('.html', '').split('_')
                        metric_name = 'unknown'
                        context_hash_short = ''
                        # Find context hash and metric name
                        event_index = parts.index('event') if 'event' in parts else -1
                        if event_index >= 0 and event_index + 1 < len(parts):
                            context_hash_short = parts[event_index + 1]  # 8-char hash
                            
                        if event_index >= 0 and len(parts) >= event_index + 4:
                            # Metric is between hash (event_index+1) and timestamp (last 2 parts)
                            metric_parts = []
                            for i in range(event_index + 2, len(parts) - 2):
                                metric_parts.append(parts[i])
                            if metric_parts:
                                metric_name = '_'.join(metric_parts)
                        
                        # Find full context_hash from contexts_info (if available)
                        # We'll search for matching context_hash that starts with context_hash_short
                        full_context_hash = context_hash_short
                        # Try to find full context_hash from events_data
                        if isinstance(events_data, list) and context_hash_short:
                            for event in events_data:
                                event_context_hash = event.get('context_hash', '')
                                if event_context_hash and event_context_hash.startswith(context_hash_short):
                                    full_context_hash = event_context_hash
                                    break
                        
                        # Get events count for this specific context + metric combination
                        context_metric_key = f"{full_context_hash}_{metric_name}"
                        metric_events = events_by_context_metric.get(context_metric_key, {'total': 0, 'positive': 0, 'negative': 0})
                        
                        visualization_files.append({
                            'path': f"/api/report/{f.name}",
                            'metric_name': metric_name,
                            'context_hash': full_context_hash,
                            'events_count': metric_events['total'],
                            'positive_events_count': metric_events['positive'],
                            'negative_events_count': metric_events['negative']
                        })
                else:
                    # Last resort: search all possible paths for any event files with job prefix
                    job_prefix = job_name.rsplit('_explore_', 1)[0] if '_explore_' in job_name else job_name
                    for search_path in all_searched_paths:
                        if not search_path.exists() or not search_path.is_dir():
                            continue
                        all_event_files = list(search_path.glob("*_event_*.html"))
                        if all_event_files:
                            prefix_files = [f for f in all_event_files if job_prefix in f.name]
                            if prefix_files:
                                prefix_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                # Create list of visualization files with metric info
                                visualization_files = []
                                for f in prefix_files:
                                    # Extract context hash and metric name from filename
                                    filename = f.name
                                    parts = filename.replace('.html', '').split('_')
                                    metric_name = 'unknown'
                                    context_hash_short = ''
                                    event_index = parts.index('event') if 'event' in parts else -1
                                    if event_index >= 0 and event_index + 1 < len(parts):
                                        context_hash_short = parts[event_index + 1]  # 8-char hash
                                        
                                    if event_index >= 0 and len(parts) >= event_index + 4:
                                        metric_parts = []
                                        for i in range(event_index + 2, len(parts) - 2):
                                            metric_parts.append(parts[i])
                                        if metric_parts:
                                            metric_name = '_'.join(metric_parts)
                                    
                                    # Find full context_hash from events_data
                                    full_context_hash = context_hash_short
                                    if isinstance(events_data, list) and context_hash_short:
                                        for event in events_data:
                                            event_context_hash = event.get('context_hash', '')
                                            if event_context_hash and event_context_hash.startswith(context_hash_short):
                                                full_context_hash = event_context_hash
                                                break
                                    
                                    # Get events count for this specific context + metric combination
                                    context_metric_key = f"{full_context_hash}_{metric_name}"
                                    metric_events = events_by_context_metric.get(context_metric_key, {'total': 0, 'positive': 0, 'negative': 0})
                                    
                                    visualization_files.append({
                                        'path': f"/api/report/{f.name}",
                                        'metric_name': metric_name,
                                        'context_hash': full_context_hash,
                                        'events_count': metric_events['total'],
                                        'positive_events_count': metric_events['positive'],
                                        'negative_events_count': metric_events['negative']
                                    })
                                break
            
            # Extract context information from events, threshold files, and visualization files
            contexts_info = []
            from collections import defaultdict
            
            # First, extract contexts from threshold files and count actual data points from source data
            contexts_by_hash = defaultdict(lambda: {
                'count': 0, 
                'context_json': '{}', 
                'metrics': set(),
                'data_points': 0  # Actual number of data points from source data
            })
            
            # Try to count actual data points from source data file
            data_points_by_context = {}
            if self.data_file:
                try:
                    import pandas as pd
                    # Resolve data file path
                    data_file_path = self.data_file
                    if not os.path.isabs(data_file_path):
                        base_path = Path(self.base_config_path).resolve()
                        project_root = base_path.parent
                        if project_root.name == 'configs':
                            project_root = project_root.parent
                        data_file_path = project_root / data_file_path
                    
                    if Path(data_file_path).exists():
                        # Load data directly using pandas (simpler, doesn't need adapter)
                        import pandas as pd
                        # Determine file type and load accordingly
                        if str(data_file_path).endswith('.parquet'):
                            df = pd.read_parquet(str(data_file_path))
                        elif str(data_file_path).endswith('.csv'):
                            df = pd.read_csv(str(data_file_path))
                        elif str(data_file_path).endswith(('.pkl', '.pickle')):
                            df = pd.read_pickle(str(data_file_path))
                        else:
                            # Try CSV as default
                            df = pd.read_csv(str(data_file_path))
                        
                        # Debug output
                        print(f"DEBUG: Loaded {len(df)} rows from {data_file_path}")
                        
                        if not df.empty:
                            print(f"DEBUG: Columns: {df.columns.tolist()}")
                            # Group by context fields only, then count all data points (across all metrics) for each context
                            context_fields = variant_config.get('context_fields', [])
                            
                            if context_fields and all(col in df.columns for col in context_fields):
                                # Group by context fields only (not by metric)
                                grouped = df.groupby(context_fields)
                                
                                # Compute context hash for each group
                                from .preprocessing import Preprocessing
                                temp_preprocessing = Preprocessing(variant_config)
                                
                                for group_key, group_df in grouped:
                                    # Extract context values from group_key
                                    # pandas groupby with multiple columns returns a tuple
                                    # The tuple order matches the order of columns in groupby()
                                    if len(context_fields) == 1:
                                        # Single field: group_key is the value directly (not a tuple)
                                        context_values = {context_fields[0]: group_key}
                                    else:
                                        # Multiple fields: group_key is ALWAYS a tuple when grouping by multiple columns
                                        # The tuple has the same length as context_fields
                                        if isinstance(group_key, tuple) and len(group_key) == len(context_fields):
                                            context_values = {context_fields[i]: group_key[i] for i in range(len(context_fields))}
                                        else:
                                            # This shouldn't happen, but handle it
                                            import warnings
                                            warnings.warn(f"Unexpected group_key format: {group_key}, type: {type(group_key)}, expected length: {len(context_fields)}")
                                            continue  # Skip this group
                                    
                                    # Compute context hash
                                    context_hash = temp_preprocessing.compute_context_hash(context_values)
                                    
                                    # Count all data points for this context (across all metrics)
                                    # This is the total number of rows (data points) for this context
                                    data_points_by_context[context_hash] = len(group_df)
                                    
                                    # Debug output
                                    print(f"DEBUG: Context {context_hash[:8]}: {len(group_df)} data points, values: {context_values}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pass
            
            # Extract context info from threshold files
            if threshold_files:
                try:
                    import pandas as pd
                    df_thresholds = pd.read_csv(threshold_files[0])
                    
                    # Group by context_hash
                    if 'context_hash' in df_thresholds.columns:
                        for _, row in df_thresholds.iterrows():
                            context_hash = str(row.get('context_hash', ''))
                            if context_hash:
                                # Try to get context_json from row or reconstruct from context fields
                                context_json = row.get('context_json', '{}')
                                if pd.isna(context_json) or not context_json:
                                    # Try to reconstruct from context fields
                                    context_dict = {}
                                    context_fields = variant_config.get('context_fields', [])
                                    for field in context_fields:
                                        if field in row:
                                            context_dict[field] = row[field]
                                    if context_dict:
                                        context_json = json.dumps(context_dict, sort_keys=True)
                                
                                metric_name = str(row.get('metric_name', ''))
                                
                                # Use actual data points count if available, otherwise count threshold rows
                                if context_hash in data_points_by_context:
                                    # Use the actual count from source data
                                    contexts_by_hash[context_hash]['data_points'] = data_points_by_context[context_hash]
                                elif contexts_by_hash[context_hash]['data_points'] == 0:
                                    # Fallback: if we don't have data from source, try to estimate from threshold file
                                    # Count unique timestamps or use row count as approximation
                                    threshold_rows_for_context = df_thresholds[df_thresholds['context_hash'] == context_hash]
                                    if not threshold_rows_for_context.empty and 'timestamp' in df_thresholds.columns:
                                        # Try to count unique timestamps if available
                                        unique_timestamps = threshold_rows_for_context['timestamp'].nunique()
                                        contexts_by_hash[context_hash]['data_points'] = unique_timestamps if unique_timestamps > 0 else len(threshold_rows_for_context)
                                    else:
                                        # Last resort: use number of threshold rows (this is number of metrics, not data points)
                                        contexts_by_hash[context_hash]['data_points'] = len(threshold_rows_for_context)
                                
                                contexts_by_hash[context_hash]['context_json'] = context_json if context_json else contexts_by_hash[context_hash]['context_json']
                                if metric_name:
                                    contexts_by_hash[context_hash]['metrics'].add(metric_name)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pass
            
            # Also extract from events if available (for context_json and events count)
            if events_data and isinstance(events_data, list):
                for event in events_data:
                    context_hash = event.get('context_hash', '')
                    context_json = event.get('context_json', '{}')
                    metric_name = event.get('metric_name', '')
                    
                    if context_hash:
                        contexts_by_hash[context_hash]['count'] += 1
                        # Use context_json from events if threshold file didn't have it
                        if not contexts_by_hash[context_hash]['context_json'] or contexts_by_hash[context_hash]['context_json'] == '{}':
                            contexts_by_hash[context_hash]['context_json'] = context_json
                        if metric_name:
                            contexts_by_hash[context_hash]['metrics'].add(metric_name)
            
            # If no data points from thresholds, count unique context_hash from visualization files
            # Each visualization file now represents one context (with all metrics)
            if all(ctx['data_points'] == 0 for ctx in contexts_by_hash.values()):
                # Count unique context_hash_short (each file is one context)
                context_hashes_seen = set()
                for viz_file in visualization_files:
                    filename = viz_file.split('/')[-1]
                    parts = filename.split('_event_')
                    if len(parts) >= 2:
                        context_hash_8chars = parts[1].split('_')[0]
                        context_hashes_seen.add(context_hash_8chars)
                
                # Distribute counts to full context hashes
                for context_hash_8chars in context_hashes_seen:
                    # Find matching full context_hash
                    for full_hash in contexts_by_hash.keys():
                        if full_hash.startswith(context_hash_8chars):
                            # Each visualization file represents one context
                            # We can't count metrics from filename anymore, but we can mark that context has data
                            if contexts_by_hash[full_hash]['data_points'] == 0:
                                contexts_by_hash[full_hash]['data_points'] = 1  # At least one visualization file exists
                            break
            
            # Convert to list and sort by data_points (number of data points) descending
            for context_hash, info in contexts_by_hash.items():
                try:
                    context_dict = json.loads(info['context_json']) if info['context_json'] else {}
                    # Format context as readable string
                    context_str = ', '.join([f"{k}={v}" for k, v in sorted(context_dict.items())])
                    if not context_str:
                        context_str = f"Context {context_hash[:8]}"
                    
                    contexts_info.append({
                        'context_hash': context_hash,
                        'context_hash_short': context_hash[:8],
                        'context_json': info['context_json'],
                        'context_display': context_str,
                        'metrics': sorted(list(info['metrics'])),
                        'events_count': info['count'],
                        'viz_count': info['data_points'],  # Actual number of data points
                    })
                except Exception:
                    continue
            
            # Sort by viz_count (number of data points) descending
            contexts_info.sort(key=lambda x: x['viz_count'], reverse=True)
            
            # Generate recommendations if no events found
            recommendations = []
            if events_count == 0:
                try:
                    from .recommendations import RecommendationGenerator
                    # Try to load data for recommendations
                    data_file_path = self.data_file
                    if data_file_path:
                        # Resolve path relative to project root if needed
                        if not os.path.isabs(data_file_path):
                            base_path = Path(self.base_config_path).resolve()
                            project_root = base_path.parent
                            if project_root.name == 'configs':
                                project_root = project_root.parent
                            data_file_path = project_root / data_file_path
                        
                        if Path(data_file_path).exists():
                            import pandas as pd
                            # Create a temporary job to access data_access
                            temp_job = AnalyticsJob(str(temp_config_path), dry_run=True, data_file=str(data_file_path))
                            df = temp_job.data_access.load_data_from_file(str(data_file_path))
                            # Get a sample series for recommendations (first metric, first context)
                            if not df.empty:
                                timestamp_field = variant_config.get('timestamp_field')
                                metric_fields = variant_config.get('metric_fields', [])
                                if timestamp_field and metric_fields:
                                    # Group by first metric
                                    metric_name_field = metric_fields[0]
                                    if metric_name_field in df.columns:
                                        first_metric = df[metric_name_field].iloc[0]
                                        metric_df = df[df[metric_name_field] == first_metric]
                                        if len(metric_df) > 0 and timestamp_field in metric_df.columns:
                                            metric_value_field = metric_fields[1] if len(metric_fields) > 1 else None
                                            if metric_value_field and metric_value_field in metric_df.columns:
                                                series = metric_df.set_index(timestamp_field)[metric_value_field].sort_index()
                                                # Use baseline result from the actual job run if available
                                                rec_baseline_result = baseline_result_for_recs
                                                if rec_baseline_result is None:
                                                    # Fallback to computed values
                                                    rec_baseline_result = {
                                                        'baseline_value': series.mean() if not series.empty else 0,
                                                        'upper_threshold': series.mean() + 2 * series.std() if not series.empty else 0,
                                                        'lower_threshold': series.mean() - 2 * series.std() if not series.empty else 0,
                                                    }
                                                recommendations = RecommendationGenerator.generate_recommendations(
                                                    series, rec_baseline_result, events_count, variant_metadata
                                                )
                except Exception as e:
                    # If recommendations fail, just continue without them
                    import traceback
                    traceback.print_exc()
                    pass
            
            return {
                'variant': variant_metadata,
                'events_count': events_count,
                'positive_events_count': positive_events_count,
                'negative_events_count': negative_events_count,
                'baselines_count': baselines_count,
                'report_path': report_path,
                'visualization_files': visualization_files,
                'contexts_info': contexts_info,  # List of contexts with counts, sorted by data points
                'recommendations': recommendations,
                'success': True
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'variant': variant_metadata,
                'events_count': 0,
                'positive_events_count': 0,
                'negative_events_count': 0,
                'baselines_count': 0,
                'report_path': None,
                'visualization_files': [],
                'recommendations': [],
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Clean up temporary config file
            try:
                if temp_config_path.exists():
                    temp_config_path.unlink()
            except:
                pass
    
    def load_and_save_data(self, output_file: Optional[str] = None) -> str:
        """
        Load data from data source and save to file for later use
        
        Args:
            output_file: Optional path to save data file (if None, auto-generate)
        
        Returns:
            Path to saved data file
        """
        from .analytics_job import AnalyticsJob
        
        # Create a temporary job to load data
        temp_job = AnalyticsJob(self.base_config_path, dry_run=True)
        
        # Load data
        print("Loading data from data source...")
        df_all = temp_job.data_access.load_measurements(start_ts=None, end_ts=None)
        
        if df_all.empty:
            raise ValueError("No data loaded from data source")
        
        # Generate output file path if not provided
        if output_file is None:
            base_path = Path(self.base_config_path).resolve()
            project_root = base_path.parent
            if project_root.name == 'configs':
                project_root = project_root.parent
            
            data_dir = project_root / 'saved_data'
            data_dir.mkdir(exist_ok=True)
            
            job_name = self.base_config.get('job', {}).get('name', 'analytics_job')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(data_dir / f"{job_name}_data_{timestamp}.parquet")
        
        # Save data
        saved_path = temp_job.data_access.save_data_to_file(df_all, output_file, format='parquet')
        print(f"Saved {len(df_all)} rows to {saved_path}")
        
        return saved_path
    
    def run_exploration(self, variant_settings: Dict[str, bool], 
                       dry_run: bool = True, 
                       max_workers: Optional[int] = None,
                       load_data_first: bool = False,
                       data_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run exploration with all variants
        
        Args:
            variant_settings: Dictionary with boolean flags for which parameters to vary
            dry_run: Whether to run in dry-run mode
            max_workers: Maximum number of parallel workers (None = auto)
            load_data_first: If True, load and save data first, then use saved data for all variants
            data_file: Optional path to saved data file (if None and load_data_first=True, auto-generate)
        
        Returns:
            List of result dictionaries, one per variant
        """
        # Load and save data first if requested
        if load_data_first and not self.data_file:
            saved_data_file = self.load_and_save_data(data_file)
            self.data_file = saved_data_file
            print(f"Using saved data file: {self.data_file}")
        
        # Clean old visualization files before running new exploration
        # Get output directory from base config
        output_dir = self.base_config.get('output', {}).get('output_dir', 'dry_run_output')
        base_path = Path(self.base_config_path).resolve()
        project_root = base_path.parent
        if project_root.name == 'configs':
            project_root = project_root.parent
        
        # Find output directory
        possible_output_paths = [
            project_root / output_dir,
            project_root / 'mass' / 'ui' / output_dir,
            Path.cwd() / output_dir,
            Path.cwd() / 'mass' / 'ui' / output_dir,
            base_path.parent / output_dir,
            base_path.parent / 'mass' / 'ui' / output_dir,
        ]
        
        job_name = self.base_config.get('job', {}).get('name', 'analytics_job')
        job_prefix = job_name.rsplit('_explore_', 1)[0] if '_explore_' in job_name else job_name
        
        # Clean old files from all possible output directories
        cleaned_count = 0
        for output_path in possible_output_paths:
            if output_path.exists() and output_path.is_dir():
                # Remove old visualization files, events, thresholds, and reports with matching prefix
                for pattern in ['*_event_*.html', '*_events_*.json', '*_thresholds_*.csv', '*_summary_*.html']:
                    for old_file in output_path.glob(pattern):
                        if job_prefix in old_file.name:
                            try:
                                old_file.unlink()
                                cleaned_count += 1
                            except Exception as e:
                                print(f"  ⚠ Could not delete {old_file.name}: {e}")
        
        if cleaned_count > 0:
            print(f"  ✓ Cleaned {cleaned_count} old files from previous exploration runs")
        
        # Generate variants
        variants = self.generate_variants(variant_settings)
        
        if not variants:
            return []
        
        # Run variants in parallel
        results = []
        if max_workers is None:
            # Limit to reasonable number to avoid overwhelming the system
            max_workers = min(len(variants), 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_variant = {
                executor.submit(self.run_variant, variant, dry_run): variant 
                for variant in variants
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_variant):
                variant = future_to_variant[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle exception
                    results.append({
                        'variant': variant.get('metadata', {}),
                        'events_count': 0,
                        'baselines_count': 0,
                        'report_path': None,
                        'success': False,
                        'error': str(e)
                    })
        
        # Sort results by variant ID to maintain order
        results.sort(key=lambda r: r.get('variant', {}).get('id', 0))
        
        return results
    
    def run_auto_tune(
        self,
        dry_run: bool = True,
        tuning_methods: Optional[List[str]] = None,
        compare_methods: bool = False
    ) -> Dict[str, Any]:
        """
        Автоматический подбор параметров и обнаружение событий
        
        Args:
            dry_run: Whether to run in dry-run mode
            tuning_methods: Список методов подбора ('adaptive', 'ml', 'hybrid').
                          Если None, используется ['adaptive']
            compare_methods: Если True, сравнивает результаты разных методов
        
        Returns:
            Dictionary with results: events, optimal_params, report_path, comparison (if compare_methods=True)
        """
        from .adaptive_parameter_tuner import (
            create_parameter_tuner,
            AdaptiveParameterTuner,
            MLParameterTuner,
            HybridParameterTuner
        )
        from .data_access import DataAccess
        from .preprocessing import Preprocessing
        from .baseline_calculator import BaselineCalculator
        from .event_detector import EventDetector
        
        if tuning_methods is None:
            tuning_methods = ['adaptive']
        
        print("=" * 80)
        print("AUTO-TUNE MODE: Automatic Parameter Tuning")
        print("=" * 80)
        
        # Очищаем старые файлы визуализации перед запуском (как в исследовании)
        print("\nStep 0: Cleaning old visualization files...")
        output_dir = self.base_config.get('output', {}).get('output_dir', 'dry_run_output')
        base_path = Path(self.base_config_path).resolve()
        project_root = base_path.parent
        if project_root.name == 'configs':
            project_root = project_root.parent
        
        # Find output directory
        possible_output_paths = [
            project_root / output_dir,
            project_root / 'mass' / 'ui' / output_dir,
            Path.cwd() / output_dir,
            Path.cwd() / 'mass' / 'ui' / output_dir,
            base_path.parent / output_dir,
            base_path.parent / 'mass' / 'ui' / output_dir,
        ]
        
        job_name = self.base_config.get('job', {}).get('name', 'auto_tune')
        job_prefix = job_name.rsplit('_explore_', 1)[0] if '_explore_' in job_name else job_name
        
        # Clean old files from all possible output directories
        cleaned_count = 0
        for output_path in possible_output_paths:
            if output_path.exists() and output_path.is_dir():
                # Remove old visualization files, events, thresholds, and reports with matching prefix
                for pattern in ['*_event_*.html', '*_events_*.json', '*_thresholds_*.csv', '*_summary_*.html']:
                    for old_file in output_path.glob(pattern):
                        if job_prefix in old_file.name:
                            try:
                                old_file.unlink()
                                cleaned_count += 1
                            except Exception as e:
                                print(f"  ⚠ Could not delete {old_file.name}: {e}")
        
        if cleaned_count > 0:
            print(f"  ✓ Cleaned {cleaned_count} old files from previous autotune runs")
        else:
            print(f"  ✓ No old files to clean")
        
        # Шаг 1: Загрузить данные
        print("\nStep 1: Loading data...")
        print(f"  DEBUG: self.data_file = {self.data_file}")
        if self.data_file:
            print(f"  DEBUG: os.path.exists(self.data_file) = {os.path.exists(self.data_file)}")
            if os.path.exists(self.data_file):
                print(f"  DEBUG: File size = {os.path.getsize(self.data_file)} bytes")
        
        job = AnalyticsJob(self.base_config_path, dry_run=dry_run, data_file=self.data_file)
        data_access = job.data_access
        preprocessing = job.preprocessing
        
        # Загружаем все данные (используем сохраненные данные если указаны)
        if self.data_file and os.path.exists(self.data_file):
            print(f"  ✓ Loading from saved data file: {self.data_file}")
            df_all = data_access.load_data_from_file(self.data_file)
        else:
            if self.data_file:
                print(f"  ⚠ Warning: data_file specified but not found: {self.data_file}")
                print("  → Falling back to loading from data source...")
            else:
                print("  → Loading from data source (no data_file specified)...")
            df_all = data_access.load_measurements()
        df_cleaned = preprocessing.clean_data(df_all, remove_outliers=True)
        grouped_data = preprocessing.group_by_context(df_cleaned)
        
        print(f"Loaded {len(grouped_data)} context×metric combinations")
        
        # Шаг 2: Создаем тюнеры
        tuners = {}
        for method in tuning_methods:
            tuners[method] = create_parameter_tuner(method)
            print(f"  Initialized {method} tuner")
        
        # Шаг 3: Анализируем каждый контекст и подбираем параметры
        print("\nStep 2: Analyzing contexts and tuning parameters...")
        print(f"  compare_methods = {compare_methods}")
        context_results = {}
        all_events = []
        all_baselines = []
        visualization_data = []  # Собираем данные для визуализации
        all_processed_contexts = []  # Сохраняем все обработанные контексты для формирования списка
        
        for group_key, group_df in grouped_data.items():
            # Извлекаем метрику и контекст
            metric_name = group_key[0] if isinstance(group_key, tuple) else group_key
            # Используем правильный метод из Preprocessing, который учитывает, что первый элемент - метрика
            context_values = preprocessing.extract_context_from_group_key(group_key)
            
            # Вычисляем context_hash для визуализации
            context_hash = preprocessing.compute_context_hash(context_values)
            
            # Конвертируем numpy типы в стандартные Python типы для JSON сериализации
            context_values_serializable = {}
            for key, value in context_values.items():
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    context_values_serializable[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    context_values_serializable[key] = float(value)
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    context_values_serializable[key] = str(value)
                elif hasattr(value, 'item'):  # другие numpy types
                    try:
                        context_values_serializable[key] = value.item()
                    except (AttributeError, ValueError):
                        context_values_serializable[key] = str(value)
                else:
                    context_values_serializable[key] = value
            
            context_json = json.dumps(context_values_serializable, sort_keys=True)
            
            # Сохраняем информацию о контексте для формирования списка
            all_processed_contexts.append({
                'context_key': str(group_key),
                'context_hash': context_hash,
                'context_json': context_json,
                'metric_name': metric_name,
                'context_values': context_values
            })
            
            # Создаем временной ряд
            series = preprocessing.prepare_time_series(group_df)
            
            # Сохраняем контекст даже если данных недостаточно (для списка контекстов)
            # Но пропускаем обработку если данных мало
            if len(series) < 3:
                print(f"  Skipping {group_key}: insufficient data ({len(series)} points)")
                # Контекст уже сохранен в all_processed_contexts выше
                continue
            
            # Анализируем характеристики данных (один раз для всех методов)
            primary_tuner = tuners[tuning_methods[0]]
            characteristics = primary_tuner.analyze_characteristics(series)
            
            # Получаем направление метрики
            metric_direction = self.base_config.get('metric_direction', {}).get(
                metric_name,
                self.base_config.get('metric_direction', {}).get('default', 'negative')
            )
            
            # Подбираем параметры каждым методом (с обработкой ошибок)
            tuning_results = {}
            tuning_errors = {}  # Сохраняем ошибки для каждого метода
            for method_name, tuner in tuners.items():
                try:
                    tuning_result = tuner.suggest_parameters(
                        characteristics, metric_name, metric_direction, str(group_key)
                    )
                    tuning_results[method_name] = tuning_result
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}"
                    tuning_errors[method_name] = error_msg
                    print(f"    ⚠ Error in {method_name} tuner for {group_key}: {error_msg}")
                    # Продолжаем с другими методами
            
            # Выбираем лучший результат (или используем все для сравнения)
            if compare_methods:
                # Сериализуем characteristics (dataclass -> dict)
                characteristics_dict = {
                    'coefficient_of_variation': characteristics.coefficient_of_variation,
                    'volatility': characteristics.volatility,
                    'trend_strength': characteristics.trend_strength,
                    'seasonality_strength': characteristics.seasonality_strength,
                    'outlier_ratio': characteristics.outlier_ratio,
                    'data_density': characteristics.data_density,
                    'baseline_stability': characteristics.baseline_stability,
                    'mean_value': characteristics.mean_value,
                    'std_value': characteristics.std_value,
                    'min_value': characteristics.min_value,
                    'max_value': characteristics.max_value,
                    'data_points': characteristics.data_points,
                }
                
                # Сериализуем tuning_results (dataclass -> dict)
                tuning_results_dict = {}
                for method_name, tuning_result in tuning_results.items():
                    tuning_results_dict[method_name] = {
                        'parameters': tuning_result.parameters,
                        'confidence': tuning_result.confidence,
                        'method': tuning_result.method,
                        'characteristics': {
                            'coefficient_of_variation': tuning_result.characteristics.coefficient_of_variation,
                            'volatility': tuning_result.characteristics.volatility,
                            'trend_strength': tuning_result.characteristics.trend_strength,
                            'seasonality_strength': tuning_result.characteristics.seasonality_strength,
                            'outlier_ratio': tuning_result.characteristics.outlier_ratio,
                            'data_density': tuning_result.characteristics.data_density,
                            'baseline_stability': tuning_result.characteristics.baseline_stability,
                            'mean_value': tuning_result.characteristics.mean_value,
                            'std_value': tuning_result.characteristics.std_value,
                            'min_value': tuning_result.characteristics.min_value,
                            'max_value': tuning_result.characteristics.max_value,
                            'data_points': tuning_result.characteristics.data_points,
                        },
                        'reasoning': tuning_result.reasoning,
                    }
                
                # Сохраняем все результаты для сравнения
                context_results[group_key] = {
                    'characteristics': characteristics_dict,
                    'tuning_results': tuning_results_dict,
                    'events_by_method': {},
                    'baselines_by_method': {}
                }
                
                # Запускаем обнаружение для каждого метода
                for method_name, tuning_result in tuning_results.items():
                    context_config = copy.deepcopy(self.base_config)
                    context_config['analytics'].update(tuning_result.parameters)
                    
                    baseline_calc = BaselineCalculator(context_config)
                    baseline_result = baseline_calc.compute_baseline_and_thresholds(series)
                    
                    event_detector = EventDetector(context_config)
                    events = event_detector.detect_events(series, baseline_result, metric_name=metric_name)
                    
                    # Сериализуем события для сравнения методов
                    events_serializable = []
                    for event in events:
                        event_serializable = {}
                        for key, value in event.items():
                            if isinstance(value, pd.Series):
                                event_serializable[key] = value.tolist()
                            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                                event_serializable[key] = str(value)
                            elif hasattr(value, 'item'):  # numpy types
                                try:
                                    event_serializable[key] = value.item()
                                except (AttributeError, ValueError):
                                    event_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                            else:
                                event_serializable[key] = value
                        # Добавляем информацию о методе для сравнения
                        event_serializable['tuning_method'] = method_name
                        event_serializable['context_hash'] = context_hash
                        event_serializable['metric_name'] = metric_name
                        events_serializable.append(event_serializable)
                    
                    # Сериализуем baseline_result для сравнения методов
                    baseline_result_serializable = {
                        'baseline_value': float(baseline_result.get('baseline_value', 0)) if baseline_result.get('baseline_value') is not None else None,
                        'upper_threshold': float(baseline_result.get('upper_threshold', 0)) if baseline_result.get('upper_threshold') is not None else None,
                        'lower_threshold': float(baseline_result.get('lower_threshold', 0)) if baseline_result.get('lower_threshold') is not None else None,
                        'baseline_method': baseline_result.get('baseline_method'),
                        'window_size': baseline_result.get('window_size'),
                        'sensitivity': float(baseline_result.get('sensitivity', 0)) if baseline_result.get('sensitivity') is not None else None,
                        'adaptive_threshold': baseline_result.get('adaptive_threshold', False),
                        'statistics': baseline_result.get('statistics', {})
                    }
                    
                    context_results[group_key]['events_by_method'][method_name] = events_serializable
                    context_results[group_key]['baselines_by_method'][method_name] = baseline_result_serializable
                    
                    # Сохраняем данные для визуализации для КАЖДОГО метода при compare_methods
                    visualization_data.append({
                        'metric_name': metric_name,
                        'context_hash': context_hash,
                        'context_json': context_json,
                        'series': series,
                        'baseline_result': baseline_result,
                        'events': events,
                        'tuning_method': method_name,
                        'tuning_confidence': tuning_result.confidence,
                        'tuning_errors': tuning_errors,  # Добавляем информацию об ошибках
                    })
                    print(f"    [DEBUG] Added visualization data for {metric_name}, context_hash={context_hash[:8]}, method={method_name}")
            else:
                # Используем первый метод (или можно выбрать лучший по confidence)
                if not tuning_results:
                    # Если все методы упали с ошибкой, пропускаем этот контекст
                    print(f"  ⚠ Skipping {group_key}: all tuning methods failed")
                    print(f"    Errors: {tuning_errors}")
                    continue
                
                best_method = max(tuning_results.keys(), key=lambda m: tuning_results[m].confidence)
                best_result = tuning_results[best_method]
                
                # Создаем конфигурацию с оптимальными параметрами
                context_config = copy.deepcopy(self.base_config)
                context_config['analytics'].update(best_result.parameters)
                
                # Вычисляем baseline
                baseline_calc = BaselineCalculator(context_config)
                baseline_result = baseline_calc.compute_baseline_and_thresholds(series)
                
                # Обнаруживаем события
                event_detector = EventDetector(context_config)
                events = event_detector.detect_events(series, baseline_result, metric_name=metric_name)
                
                # Сохраняем результаты (сериализуем события)
                for event in events:
                    # Сериализуем event (убираем любые Series/numpy объекты)
                    event_serializable = {}
                    for key, value in event.items():
                        if isinstance(value, pd.Series):
                            event_serializable[key] = value.tolist()
                        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                            event_serializable[key] = str(value)
                        elif hasattr(value, 'item'):  # numpy types
                            try:
                                event_serializable[key] = value.item()
                            except (AttributeError, ValueError):
                                event_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                        else:
                            event_serializable[key] = value
                    
                    # Сериализуем optimal_params (может содержать Series/numpy типы)
                    optimal_params_serializable = {}
                    for key, value in best_result.parameters.items():
                        if isinstance(value, pd.Series):
                            optimal_params_serializable[key] = value.tolist()
                        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                            optimal_params_serializable[key] = str(value)
                        elif hasattr(value, 'item'):  # numpy types
                            try:
                                optimal_params_serializable[key] = value.item()
                            except (AttributeError, ValueError):
                                optimal_params_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                        else:
                            optimal_params_serializable[key] = value
                    
                    event_serializable['context_key'] = str(group_key)
                    event_serializable['context_hash'] = context_hash  # Добавляем context_hash для сопоставления
                    event_serializable['context_json'] = context_json  # Добавляем context_json для отображения
                    event_serializable['metric_name'] = metric_name  # Добавляем metric_name для сопоставления
                    event_serializable['optimal_params'] = optimal_params_serializable
                    event_serializable['tuning_method'] = best_method
                    event_serializable['tuning_confidence'] = best_result.confidence
                    all_events.append(event_serializable)
                
                # Сериализуем baseline_result (убираем Series объекты)
                baseline_result_serializable = {
                    'baseline_value': float(baseline_result.get('baseline_value', 0)) if baseline_result.get('baseline_value') is not None else None,
                    'upper_threshold': float(baseline_result.get('upper_threshold', 0)) if baseline_result.get('upper_threshold') is not None else None,
                    'lower_threshold': float(baseline_result.get('lower_threshold', 0)) if baseline_result.get('lower_threshold') is not None else None,
                    'baseline_method': baseline_result.get('baseline_method'),
                    'window_size': baseline_result.get('window_size'),
                    'sensitivity': float(baseline_result.get('sensitivity', 0)) if baseline_result.get('sensitivity') is not None else None,
                    'adaptive_threshold': baseline_result.get('adaptive_threshold', False),
                    'statistics': baseline_result.get('statistics', {})
                }
                
                # Сериализуем optimal_params для baselines
                optimal_params_serializable = {}
                for key, value in best_result.parameters.items():
                    if isinstance(value, pd.Series):
                        optimal_params_serializable[key] = value.tolist()
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        optimal_params_serializable[key] = str(value)
                    elif hasattr(value, 'item'):  # numpy types
                        try:
                            optimal_params_serializable[key] = value.item()
                        except (AttributeError, ValueError):
                            optimal_params_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                    else:
                        optimal_params_serializable[key] = value
                
                all_baselines.append({
                    'context_key': str(group_key),
                    'baseline_result': baseline_result_serializable,
                    'optimal_params': optimal_params_serializable,
                    'tuning_method': best_method,
                    'tuning_confidence': best_result.confidence,
                    'reasoning': best_result.reasoning,
                })
                
                # Сохраняем данные для визуализации
                visualization_data.append({
                    'metric_name': metric_name,
                    'context_hash': context_hash,
                    'context_json': context_json,
                    'series': series,
                    'baseline_result': baseline_result,
                    'events': events,
                    'tuning_method': best_method,
                    'tuning_confidence': best_result.confidence,
                    'tuning_errors': tuning_errors,  # Добавляем информацию об ошибках
                })
                print(f"    [DEBUG] Added visualization data for {metric_name}, context_hash={context_hash[:8]}, events={len(events)}, method={best_method}")
                
                print(f"  {group_key}: {len(events)} events, method={best_method}, "
                      f"params={best_result.parameters.get('window_size')}/{best_result.parameters.get('sensitivity')}")
        
        # Шаг 4: Генерируем визуализации
        visualization_files = []
        print(f"\n[DEBUG] visualization_data length: {len(visualization_data)}")
        if visualization_data:
            print(f"\nStep 3: Generating visualizations... (found {len(visualization_data)} data entries)")
            try:
                from .visualization import EventVisualizer
                from collections import defaultdict
                from datetime import datetime
                
                # Группируем по context_hash (и tuning_method при compare_methods)
                visualization_by_context = defaultdict(list)
                for viz_data in visualization_data:
                    context_hash = viz_data.get('context_hash', '')
                    # При compare_methods создаем отдельные группы для каждого метода
                    if compare_methods:
                        tuning_method = viz_data.get('tuning_method', 'unknown')
                        # Используем комбинированный ключ: context_hash + method
                        group_key = f"{context_hash}_{tuning_method}"
                    else:
                        group_key = context_hash
                    visualization_by_context[group_key].append(viz_data)
                
                # Добавляем все события для каждого контекста (как в исследовании)
                for group_key, metrics_data in visualization_by_context.items():
                    for metric_data in metrics_data:
                        metric_name = metric_data.get('metric_name', '')
                        context_hash = metric_data.get('context_hash', '')
                        tuning_method = metric_data.get('tuning_method', '')
                        # Находим все события для этого context_hash, metric_name и tuning_method
                        matching_events = [
                            event for event in all_events
                            if event.get('context_hash') == context_hash 
                            and event.get('metric_name') == metric_name
                            and (not compare_methods or event.get('tuning_method') == tuning_method)
                        ]
                        # Заменяем события на все найденные
                        metric_data['events'] = matching_events
                
                print(f"  Grouped into {len(visualization_by_context)} contexts" + 
                      (f" ({len(tuning_methods)} methods per context)" if compare_methods else ""))
                
                # Генерируем визуализации
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                job_name = self.base_config.get('job', {}).get('name', 'auto_tune')
                output_dir = self.base_config.get('output', {}).get('output_dir', 'dry_run_output')
                
                print(f"  Output directory: {output_dir}")
                print(f"  Job name: {job_name}")
                print(f"  Timestamp: {timestamp}")
                
                os.makedirs(output_dir, exist_ok=True)
                
                EventVisualizer.generate_visualizations(
                    visualization_by_context,
                    output_dir,
                    timestamp,
                    job_name
                )
                
                # Находим сгенерированные файлы (только с текущим timestamp)
                output_path = Path(output_dir)
                if output_path.exists():
                    # Ищем файлы по паттерну: {job_name}_event_{hash}_{metric}_{timestamp}.html
                    # Фильтруем только файлы с текущим timestamp
                    viz_pattern = f"{job_name}_event_*.html"
                    all_found_files = list(output_path.glob(viz_pattern))
                    
                    # Фильтруем только файлы с текущим timestamp
                    found_files = [f for f in all_found_files if timestamp in f.name]
                    print(f"  Found {len(found_files)} visualization files with current timestamp '{timestamp}' (total matching pattern: {len(all_found_files)})")
                    
                    # Если не нашли, пробуем более широкий паттерн, но все равно фильтруем по timestamp
                    if not found_files:
                        viz_pattern_wide = "*_event_*.html"
                        all_found_files_wide = list(output_path.glob(viz_pattern_wide))
                        found_files = [f for f in all_found_files_wide if timestamp in f.name]
                        print(f"  Trying wider pattern '{viz_pattern_wide}': found {len(found_files)} files with current timestamp (total: {len(all_found_files_wide)})")
                    
                    for f in found_files:
                        # Извлекаем context_hash из имени файла
                        # Формат: {job_name}_event_{hash}_{metric_or_all_metrics}_{timestamp}.html
                        filename = f.name
                        print(f"    Processing file: {filename}")
                        parts = filename.replace('.html', '').split('_')
                        context_hash_short = ''
                        event_index = parts.index('event') if 'event' in parts else -1
                        if event_index >= 0 and event_index + 1 < len(parts):
                            context_hash_short = parts[event_index + 1]
                        
                        print(f"    Extracted: context_hash={context_hash_short}")
                        
                        # Находим полный context_hash
                        full_context_hash = context_hash_short
                        for viz_data in visualization_data:
                            if viz_data.get('context_hash', '').startswith(context_hash_short):
                                full_context_hash = viz_data.get('context_hash', '')
                                break
                        
                        # Находим все метрики для этого контекста
                        context_metrics = set()
                        for viz_data in visualization_data:
                            if viz_data.get('context_hash', '') == full_context_hash or \
                               (viz_data.get('context_hash', '').startswith(context_hash_short)):
                                metric_name = viz_data.get('metric_name', '')
                                if metric_name:
                                    context_metrics.add(metric_name)
                        
                        # Подсчитываем ВСЕ события для этого контекста (все метрики)
                        events_count = 0
                        positive_events = 0
                        negative_events = 0
                        for event in all_events:
                            event_context_hash = event.get('context_hash', '')
                            # Сравниваем по context_hash (полному или короткому)
                            if full_context_hash and \
                               (event_context_hash == full_context_hash or 
                                event_context_hash.startswith(context_hash_short)):
                                events_count += 1
                                event_type = event.get('event_type', '')
                                if 'improvement' in event_type:
                                    positive_events += 1
                                elif 'degradation' in event_type:
                                    negative_events += 1
                        
                        # Используем первую метрику или 'all_metrics' если несколько
                        metric_name = sorted(list(context_metrics))[0] if context_metrics else 'unknown'
                        if len(context_metrics) > 1:
                            metric_name = 'all_metrics'
                        
                        # Находим информацию о методе и ошибках из visualization_data
                        tuning_method = 'unknown'
                        tuning_confidence = 0
                        tuning_errors = {}
                        for viz_data in visualization_data:
                            if viz_data.get('context_hash', '') == full_context_hash or \
                               (viz_data.get('context_hash', '').startswith(context_hash_short)):
                                tuning_method = viz_data.get('tuning_method', 'unknown')
                                tuning_confidence = viz_data.get('tuning_confidence', 0)
                                tuning_errors = viz_data.get('tuning_errors', {})
                                break
                        
                        visualization_files.append({
                            'path': f"/api/report/{f.name}",
                            'metric_name': metric_name,  # Может быть 'all_metrics' если несколько метрик
                            'context_hash': full_context_hash,
                            'events_count': events_count,  # Все события для контекста
                            'positive_events_count': positive_events,
                            'negative_events_count': negative_events,
                            'tuning_method': tuning_method,
                            'tuning_confidence': tuning_confidence,
                            'tuning_errors': tuning_errors
                        })
                
                print(f"  Generated {len(visualization_files)} visualization files")
            except Exception as e:
                import traceback
                print(f"  ⚠ Error generating visualizations: {e}")
                traceback.print_exc()
        
        # Собираем информацию о контекстах (как в исследовании)
        contexts_info = []
        from collections import defaultdict
        contexts_by_hash = defaultdict(lambda: {
            'count': 0,
            'context_json': '{}',
            'context_key': None,  # Сохраняем context_key для отображения
            'metrics': set(),
            'data_points': 0
        })
        
        # Сначала собираем ВСЕ контексты из all_processed_contexts (там есть все проанализированные контексты)
        for ctx_info in all_processed_contexts:
            context_hash = ctx_info.get('context_hash', '')
            context_json = ctx_info.get('context_json', '{}')
            context_key_str = ctx_info.get('context_key', '')
            metric_name = ctx_info.get('metric_name', '')
            
            if context_hash:
                contexts_by_hash[context_hash]['context_key'] = context_key_str
                contexts_by_hash[context_hash]['context_json'] = context_json
                if metric_name:
                    contexts_by_hash[context_hash]['metrics'].add(metric_name)
        
        print(f"  Collected {len(contexts_by_hash)} unique contexts from {len(all_processed_contexts)} processed groups")
        
        # Подсчитываем события по контекстам
        for event in all_events:
            context_hash = event.get('context_hash', '')
            context_json = event.get('context_json', '{}')
            metric_name = event.get('metric_name', '')
            
            if context_hash:
                contexts_by_hash[context_hash]['count'] += 1
                if context_json and context_json != '{}':
                    contexts_by_hash[context_hash]['context_json'] = context_json
                if metric_name:
                    contexts_by_hash[context_hash]['metrics'].add(metric_name)
        
        # Подсчитываем количество графиков (визуализаций) для каждого контекста
        # Теперь один график на контекст, поэтому просто отмечаем что есть визуализация
        for viz_file in visualization_files:
            context_hash = viz_file.get('context_hash', '')
            metric_name = viz_file.get('metric_name', '')
            if context_hash:
                contexts_by_hash[context_hash]['data_points'] = 1  # Один график на контекст
                if metric_name and metric_name != 'all_metrics':
                    contexts_by_hash[context_hash]['metrics'].add(metric_name)
                elif metric_name == 'all_metrics':
                    # Если all_metrics, добавляем все метрики из visualization_data
                    for viz_data in visualization_data:
                        if viz_data.get('context_hash', '') == context_hash:
                            m_name = viz_data.get('metric_name', '')
                            if m_name:
                                contexts_by_hash[context_hash]['metrics'].add(m_name)
        
        # Формируем список контекстов
        for context_hash, info in contexts_by_hash.items():
            try:
                context_dict = json.loads(info['context_json']) if info['context_json'] else {}
                
                # Форматируем контекст как читаемую строку
                context_str = ', '.join([f"{k}={v}" for k, v in sorted(context_dict.items())])
                
                # Если нет context_json, пытаемся использовать context_key
                if not context_str and info.get('context_key'):
                    # Форматируем context_key для отображения
                    context_key_str = info['context_key']
                    # Убираем лишние символы и форматируем
                    if context_key_str.startswith('(') and context_key_str.endswith(')'):
                        # Парсим tuple и форматируем
                        try:
                            import ast
                            context_key = ast.literal_eval(context_key_str)
                            if isinstance(context_key, tuple) and len(context_key) > 1:
                                # Пропускаем первый элемент (метрику), остальное - контекст
                                context_parts = []
                                for i, val in enumerate(context_key[1:], 1):
                                    # Пытаемся определить имя поля из конфига
                                    context_fields = self.base_config.get('context_fields', [])
                                    if i <= len(context_fields):
                                        field_name = context_fields[i-1]
                                    else:
                                        field_name = f"field{i}"
                                    context_parts.append(f"{field_name}={val}")
                                context_str = ', '.join(context_parts)
                        except Exception:
                            context_str = context_key_str
                    else:
                        context_str = context_key_str
                
                if not context_str:
                    context_str = f"Context {context_hash[:8]}"
                
                contexts_info.append({
                    'context_hash': context_hash,
                    'context_hash_short': context_hash[:8],
                    'context_json': info['context_json'],
                    'context_display': context_str,
                    'metrics': sorted(list(info['metrics'])),
                    'events_count': info['count'],
                    'viz_count': info['data_points'],  # Количество графиков для этого контекста
                })
            except Exception as e:
                import traceback
                print(f"  ⚠ Error processing context {context_hash[:8]}: {e}")
                continue
        
        # Сортируем по количеству событий (убывание)
        contexts_info.sort(key=lambda x: x['events_count'], reverse=True)
        
        # Шаг 5: Генерируем отчет
        print("\nStep 4: Generating report...")
        
        result = {
            'success': True,
            'events': all_events,
            'events_count': len(all_events),
            'contexts_analyzed': len(context_results) if compare_methods else len(all_baselines),
            'baselines': all_baselines,
            'visualization_files': visualization_files,  # Добавляем пути к файлам визуализации
            'contexts_info': contexts_info,  # Добавляем информацию о контекстах
        }
        
        if compare_methods:
            result['comparison'] = self._compare_tuning_methods(context_results)
        
        # Сохраняем оптимальные параметры (уже сериализованы в all_baselines)
        if not compare_methods:
            optimal_params_summary = {}
            for baseline in all_baselines:
                context_key = baseline['context_key']
                optimal_params_summary[context_key] = {
                    'parameters': baseline['optimal_params'],  # Уже сериализованы
                    'method': baseline['tuning_method'],
                    'confidence': baseline['tuning_confidence'],
                    'reasoning': baseline.get('reasoning', ''),
                }
            result['optimal_params'] = optimal_params_summary
        
        print(f"\n{'='*80}")
        print(f"Auto-tune completed: {len(all_events)} events found in {len(all_baselines)} contexts")
        print(f"{'='*80}")
        
        return result
    
    def _compare_tuning_methods(self, context_results: Dict) -> Dict[str, Any]:
        """Сравнение результатов разных методов подбора"""
        if not context_results:
            return {
                'methods': [],
                'contexts': {},
                'summary': {}
            }
        
        first_context = next(iter(context_results.values()))
        comparison = {
            'methods': list(first_context['tuning_results'].keys()) if 'tuning_results' in first_context else [],
            'contexts': {},
            'summary': {}
        }
        
        method_stats = {method: {'events': 0, 'avg_confidence': 0.0} for method in comparison['methods']}
        
        for context_key, context_data in context_results.items():
            # characteristics уже сериализован в dict
            characteristics = context_data.get('characteristics', {})
            context_comparison = {
                'characteristics': {
                    'cv': characteristics.get('coefficient_of_variation', 0),
                    'stability': characteristics.get('baseline_stability', 0),
                    'volatility': characteristics.get('volatility', 0),
                },
                'tuning_results': {},
                'events_by_method': {}
            }
            
            # tuning_results уже сериализованы в dict
            for method_name, tuning_result_dict in context_data.get('tuning_results', {}).items():
                # Сериализуем parameters если нужно
                parameters_serializable = {}
                for key, value in tuning_result_dict.get('parameters', {}).items():
                    if isinstance(value, pd.Series):
                        parameters_serializable[key] = value.tolist()
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        parameters_serializable[key] = str(value)
                    elif hasattr(value, 'item'):  # numpy types
                        try:
                            parameters_serializable[key] = value.item()
                        except (AttributeError, ValueError):
                            parameters_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                    else:
                        parameters_serializable[key] = value
                
                context_comparison['tuning_results'][method_name] = {
                    'parameters': parameters_serializable,
                    'confidence': tuning_result_dict.get('confidence', 0),
                    'reasoning': tuning_result_dict.get('reasoning', ''),
                }
                
                events = context_data.get('events_by_method', {}).get(method_name, [])
                context_comparison['events_by_method'][method_name] = len(events)
                method_stats[method_name]['events'] += len(events)
            
            # Вычисляем среднюю уверенность для каждого метода
            for method_name in comparison['methods']:
                confidences = [
                    r.get('confidence', 0) for r in context_data.get('tuning_results', {}).values()
                    if r.get('method') == method_name
                ]
                if confidences:
                    method_stats[method_name]['avg_confidence'] += sum(confidences) / len(confidences)
            
            comparison['contexts'][str(context_key)] = context_comparison
        
        # Нормализуем статистику
        num_contexts = len(context_results)
        for method_name in comparison['methods']:
            if num_contexts > 0:
                method_stats[method_name]['avg_confidence'] /= num_contexts
        
        comparison['summary'] = method_stats
        
        return comparison

