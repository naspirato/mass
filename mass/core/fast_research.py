#!/usr/bin/env python3
"""
Fast Research module for testing anomaly detection tools out-of-the-box
Supports: PyOD, alibi-detect, dtaianomaly, sklearn
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class FastResearchRunner:
    """Run fast research with multiple anomaly detection tools"""
    
    def __init__(self, config_path: str, data_file: Optional[str] = None):
        """
        Initialize Fast Research Runner
        
        Args:
            config_path: Path to config YAML file
            data_file: Optional path to saved data file
        """
        from mass.core.config_loader import ConfigLoader
        from mass.core.data_access import DataAccess
        from mass.core.analytics_job import create_adapter
        
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Create data access
        adapter = create_adapter(self.config)
        self.data_access = DataAccess(adapter, self.config)
        
        self.data_file = data_file
        self.tools = self._load_tools()
    
    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load tools from tools.txt file"""
        tools_file = Path(__file__).parent.parent.parent / 'configs' / 'tools.txt'
        tools = []
        
        if not tools_file.exists():
            # Default tools if file doesn't exist
            return [
                {'name': 'PyOD', 'description': 'PyOD - Simple interface for anomaly detection'},
                {'name': 'alibi-detect', 'description': 'alibi-detect - Out-of-the-box outlier detection'},
                {'name': 'dtaianomaly', 'description': 'dtaianomaly - Time series anomaly detection'},
                {'name': 'Scikit-learn', 'description': 'Scikit-learn - IsolationForest for outliers'}
            ]
        
        with open(tools_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse tools from file
        # Format: tool name on its own line, then description lines, then empty line
        lines = content.split('\n')
        current_tool = None
        current_description = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip code blocks (lines starting with python or containing backticks)
            if line_stripped.startswith('python') or '`' in line_stripped:
                continue
            
            # Empty line - end of current tool section
            if not line_stripped:
                if current_tool:
                    tools.append({
                        'name': current_tool,
                        'description': '\n'.join(current_description).strip()
                    })
                    current_tool = None
                    current_description = []
                continue
            
            # Check if this is a tool name
            # Tool names are: short lines (usually < 50 chars), start with capital letter or have dash
            # Common patterns: "PyOD", "alibi-detect", "dtaianomaly", "Scikit-learn (для простых случаев)"
            is_tool_name = False
            tool_name = line_stripped
            
            # Remove parentheses content for tool name detection
            if '(' in tool_name:
                tool_name = tool_name.split('(')[0].strip()
            
            # Check if it looks like a tool name
            if (len(tool_name) < 50 and 
                (tool_name[0].isupper() or tool_name.lower().startswith('alibi') or tool_name.lower().startswith('dtaianomaly')) and
                not any(word in line_stripped.lower() for word in ['очень', 'простой', 'интерфейс:', 'пример:', 'специализация:', 'ориентирован', 'если уже'])):
                is_tool_name = True
            
            if is_tool_name and (i == 0 or not lines[i-1].strip() or current_tool is None):
                # Save previous tool if exists
                if current_tool:
                    tools.append({
                        'name': current_tool,
                        'description': '\n'.join(current_description).strip()
                    })
                # Start new tool - use cleaned name (without parentheses)
                current_tool = tool_name
                current_description = []
            else:
                # Add to description
                if current_tool:
                    current_description.append(line)
                # If no tool yet, this might be the first tool name
                elif not current_tool and len(tool_name) < 50 and (tool_name[0].isupper() or tool_name.lower().startswith('alibi')):
                    current_tool = tool_name
                    current_description = []
        
        # Add last tool if exists
        if current_tool:
            tools.append({
                'name': current_tool,
                'description': '\n'.join(current_description).strip()
            })
        
        # If no tools found, use defaults
        if not tools:
            tools = [
                {'name': 'PyOD', 'description': 'PyOD - Simple interface for anomaly detection'},
                {'name': 'alibi-detect', 'description': 'alibi-detect - Out-of-the-box outlier detection'},
                {'name': 'dtaianomaly', 'description': 'dtaianomaly - Time series anomaly detection'},
                {'name': 'Scikit-learn', 'description': 'Scikit-learn - IsolationForest for outliers'}
            ]
        
        # TEMPORARY: Only use PyOD for testing
        tools = [t for t in tools if t['name'] == 'PyOD']
        if not tools:
            # Fallback if PyOD not found
            tools = [{'name': 'PyOD', 'description': 'PyOD - Simple interface for anomaly detection'}]
        
        return tools
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file or data source"""
        if self.data_file:
            # Load from saved file
            data_file_path = Path(self.data_file)
            if not data_file_path.is_absolute():
                data_file = Path(__file__).parent.parent.parent / self.data_file
            else:
                data_file = data_file_path
            
            # Load data directly without validation (we validate ourselves)
            import pandas as pd
            
            if data_file.suffix == '.csv':
                df = pd.read_csv(data_file)
            elif data_file.suffix == '.parquet':
                df = pd.read_parquet(data_file)
            elif data_file.suffix in ['.pkl', '.pickle']:
                import pickle
                with open(data_file, 'rb') as f:
                    df = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {data_file.suffix}. Use .parquet, .csv, or .pkl")
            
            # Basic validation: check that required fields from config exist
            timestamp_field = self.config.get('timestamp_field', 'timestamp')
            metric_fields = self.config.get('metric_fields', [])
            context_fields = self.config.get('context_fields', [])
            
            required_fields = [timestamp_field]
            if metric_fields and len(metric_fields) >= 2:
                required_fields.extend(metric_fields[:2])  # metric_name and metric_value
            required_fields.extend(context_fields)
            
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in data file: {', '.join(missing_fields)}. "
                    f"Available fields: {', '.join(df.columns)}. "
                    f"Please check your config: timestamp_field={timestamp_field}, metric_fields={metric_fields}, context_fields={context_fields}"
                )
            
            # Sort by timestamp if available
            if timestamp_field in df.columns:
                df = df.sort_values(by=timestamp_field).reset_index(drop=True)
            
            return df
        else:
            # Load from data source
            return self.data_access.load_measurements()
    
    def _get_context_hash(self, row: pd.Series, context_fields: List[str]) -> str:
        """Generate hash for context"""
        context_values = [str(row[field]) for field in context_fields if field in row]
        context_str = '|'.join(context_values)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _get_context_display(self, row: pd.Series, context_fields: List[str]) -> str:
        """Generate display string for context"""
        context_parts = []
        for field in context_fields:
            if field in row:
                context_parts.append(f"{field}={row[field]}")
        return ', '.join(context_parts)
    
    def _remove_outliers(self, series: pd.Series, method: str = 'iqr') -> Tuple[pd.Series, np.ndarray]:
        """
        Remove outliers from time series data before trend change detection
        
        Args:
            series: Time series data
            method: Method for outlier removal ('iqr', 'zscore', 'isolation')
            
        Returns:
            Tuple of (cleaned_series, outlier_mask) where outlier_mask is boolean array
        """
        if len(series) < 4:
            # Too few points, return as is
            return series, np.zeros(len(series), dtype=bool)
        
        values_array = series.values
        outlier_mask = np.zeros(len(values_array), dtype=bool)
        
        if method == 'iqr':
            # IQR method: remove points outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
            Q1 = np.percentile(values_array, 25)
            Q3 = np.percentile(values_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
        elif method == 'zscore':
            # Z-score method: remove points with |z-score| > 3
            z_scores = np.abs((values_array - np.mean(values_array)) / (np.std(values_array) + 1e-6))
            outlier_mask = z_scores > 3
        elif method == 'isolation':
            # Isolation Forest method
            try:
                from sklearn.ensemble import IsolationForest
                if len(values_array) > 10:
                    model = IsolationForest(contamination=0.1, random_state=42)
                    predictions = model.fit_predict(values_array.reshape(-1, 1))
                    outlier_mask = predictions == -1
            except ImportError:
                # Fallback to IQR if sklearn not available
                Q1 = np.percentile(values_array, 25)
                Q3 = np.percentile(values_array, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
        
        # Don't remove more than 10% of data
        outlier_count = np.sum(outlier_mask)
        if outlier_count > len(values_array) * 0.1:
            # Keep only top 10% most extreme outliers
            if method == 'iqr':
                # Recalculate with stricter bounds
                Q1 = np.percentile(values_array, 25)
                Q3 = np.percentile(values_array, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.0 * IQR  # Stricter
                upper_bound = Q3 + 2.0 * IQR
                outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((values_array - np.mean(values_array)) / (np.std(values_array) + 1e-6))
                outlier_mask = z_scores > 3.5  # Stricter
        
        # Return cleaned series (outliers replaced by interpolation)
        cleaned_series = series.copy()
        cleaned_series.iloc[outlier_mask] = np.nan
        # Use bfill/ffill for pandas compatibility
        cleaned_series = cleaned_series.bfill().ffill()
        # Fallback to interpolation if still has NaN
        if cleaned_series.isna().any():
            cleaned_series = cleaned_series.interpolate(method='linear', limit_direction='both')
        
        return cleaned_series, outlier_mask
    
    def _prepare_data_for_tool(self, series: pd.Series, remove_outliers: bool = True) -> np.ndarray:
        """Prepare time series data for anomaly detection tools"""
        # Remove outliers first if requested
        if remove_outliers and len(series) >= 4:
            series, _ = self._remove_outliers(series, method='iqr')
        
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return np.array([]).reshape(0, 1)
        
        # Get values
        values = series_clean.values
        
        # Ensure values are numeric
        values = pd.to_numeric(values, errors='coerce')
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            return np.array([]).reshape(0, 1)
        
        # For time series anomaly detection, create features:
        # 1. Original values
        # 2. Rolling statistics (if enough data)
        # 3. Difference from rolling mean
        
        if len(values) > 3:
            # Create rolling features
            values_series = pd.Series(values)
            window = min(5, max(3, len(values) // 3))
            
            rolling_mean = values_series.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = values_series.rolling(window=window, center=True, min_periods=1).std().fillna(0)
            
            # Fill any remaining NaN
            rolling_mean = rolling_mean.bfill().ffill()
            
            # Create features: value, deviation from mean, normalized deviation
            deviation = values - rolling_mean.values
            normalized_deviation = np.divide(deviation, rolling_std.values + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Combine features
            X = np.column_stack([values, deviation, normalized_deviation])
        else:
            # Too few points, just use values in 2D format
            X = values.reshape(-1, 1)
        
        # Remove any NaN that might have been introduced
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) == 0:
            return np.array([]).reshape(0, 1)
        
        return X
    
    def _detect_changepoints_adtk(self, series: pd.Series) -> Optional[List[Tuple[int, int]]]:
        """
        Detect changepoints (trend changes) using ADTK LevelShiftAD
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        try:
            from adtk.detector import LevelShiftAD
            import warnings
            from contextlib import redirect_stdout, redirect_stderr
            from io import StringIO
            
            if len(series) < 10:
                return None
            
            # Suppress output
            dummy_stdout = StringIO()
            dummy_stderr = StringIO()
            
            with warnings.catch_warnings(), redirect_stdout(dummy_stdout), redirect_stderr(dummy_stderr):
                warnings.simplefilter('ignore')
                
                # Use LevelShiftAD to detect level shifts
                window = min(10, max(5, len(series) // 10))
                detector = LevelShiftAD(c=3.0, side='both', window=window)
                anomalies = detector.fit_detect(series)
                
                if anomalies is None or len(anomalies) == 0:
                    return None
                
                # Convert boolean series to intervals
                intervals = []
                in_interval = False
                start_idx = None
                
                for i, is_anomaly in enumerate(anomalies):
                    if is_anomaly and not in_interval:
                        # Start of new interval
                        start_idx = i
                        in_interval = True
                    elif not is_anomaly and in_interval:
                        # End of interval
                        intervals.append((start_idx, i - 1))
                        in_interval = False
                
                # Handle case where interval extends to end
                if in_interval:
                    intervals.append((start_idx, len(anomalies) - 1))
                
                return intervals if intervals else None
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def _detect_changepoints_rolling(self, series: pd.Series, window_size: int = None) -> Optional[List[Tuple[int, int]]]:
        """
        Detect changepoints using rolling window comparison
        
        Compares mean values in adjacent windows to detect significant changes
        """
        if len(series) < 20:
            return None
        
        if window_size is None:
            window_size = min(10, max(5, len(series) // 10))
        
        values = series.values
        intervals = []
        
        # Compare adjacent windows
        for i in range(window_size, len(values) - window_size, window_size // 2):
            window1 = values[i - window_size:i]
            window2 = values[i:i + window_size]
            
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)
            std1 = np.std(window1) + 1e-6
            std2 = np.std(window2) + 1e-6
            
            # Calculate z-score for difference
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            z_score = abs(mean2 - mean1) / (pooled_std + 1e-6)
            
            # Significant change if z-score > 2.5 (more strict)
            if z_score > 2.5:
                # Mark smaller interval around changepoint (only around the actual change point)
                start_idx = max(0, i - window_size // 4)  # Smaller interval
                end_idx = min(len(values) - 1, i + window_size // 4)
                intervals.append((start_idx, end_idx))
        
        # Merge overlapping intervals
        if not intervals:
            return None
        
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + window_size:
                # Merge intervals
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged if merged else None
    
    def _detect_changepoints_cusum(self, series: pd.Series) -> Optional[List[Tuple[int, int]]]:
        """
        Detect changepoints using CUSUM (Cumulative Sum) method
        """
        if len(series) < 20:
            return None
        
        values = series.values
        mean = np.mean(values)
        std = np.std(values) + 1e-6
        
        # Calculate CUSUM
        s_pos = np.zeros(len(values))
        s_neg = np.zeros(len(values))
        
        for i in range(1, len(values)):
            s_pos[i] = max(0, s_pos[i-1] + (values[i] - mean) / std - 0.5)
            s_neg[i] = max(0, s_neg[i-1] - (values[i] - mean) / std - 0.5)
        
        # Detect changepoints where CUSUM exceeds threshold
        threshold = 5.0  # Adjust based on data
        changepoints = []
        
        for i in range(1, len(values)):
            if s_pos[i] > threshold or s_neg[i] > threshold:
                changepoints.append(i)
        
        if not changepoints:
            return None
        
        # Convert changepoints to intervals
        intervals = []
        window = min(5, len(values) // 20)
        
        for cp in changepoints:
            start_idx = max(0, cp - window)
            end_idx = min(len(values) - 1, cp + window)
            intervals.append((start_idx, end_idx))
        
        # Merge overlapping intervals
        if not intervals:
            return None
        
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 5:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged if merged else None
    
    def _detect_trend_changes(self, series: pd.Series) -> List[Tuple[int, int]]:
        """
        Detect trend changes using multiple methods and combine results
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        # Define maximum interval size (10% of data, but at least 10 points)
        max_interval_size = max(10, len(series) // 10)
        
        def limit_interval_size(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Limit each interval to max_interval_size, split if too large"""
            if not intervals:
                return []
            limited = []
            for start, end in intervals:
                interval_size = end - start + 1
                if interval_size <= max_interval_size:
                    limited.append((start, end))
                else:
                    # Split large interval into multiple smaller ones
                    num_chunks = (interval_size + max_interval_size - 1) // max_interval_size
                    chunk_size = interval_size // num_chunks
                    
                    for i in range(num_chunks):
                        chunk_start = start + i * chunk_size
                        chunk_end = min(start + (i + 1) * chunk_size - 1, end)
                        if chunk_end >= chunk_start:  # Valid interval
                            limited.append((chunk_start, chunk_end))
            return limited
        
        all_intervals = []
        
        # Try ADTK LevelShiftAD - limit size BEFORE adding
        adtk_intervals = self._detect_changepoints_adtk(series)
        if adtk_intervals:
            all_intervals.extend(limit_interval_size(adtk_intervals))
        
        # Try rolling window comparison - limit size BEFORE adding
        rolling_intervals = self._detect_changepoints_rolling(series)
        if rolling_intervals:
            all_intervals.extend(limit_interval_size(rolling_intervals))
        
        # Try CUSUM - limit size BEFORE adding
        cusum_intervals = self._detect_changepoints_cusum(series)
        if cusum_intervals:
            all_intervals.extend(limit_interval_size(cusum_intervals))
        
        if not all_intervals:
            return []
        
        # Merge overlapping intervals from all methods (strict merging)
        all_intervals = sorted(all_intervals, key=lambda x: x[0])
        merged = [all_intervals[0]]
        
        for current in all_intervals[1:]:
            last = merged[-1]
            # Merge only if intervals overlap or are very close (within 2 points)
            if current[0] <= last[1] + 2:
                # Check if merged interval would be too large
                merged_size = max(last[1], current[1]) - last[0] + 1
                if merged_size <= max_interval_size:
                    # Safe to merge
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    # Don't merge, keep separate
                    merged.append(current)
            else:
                merged.append(current)
        
        # Final pass: split any remaining oversized intervals
        final_intervals = []
        for start, end in merged:
            interval_size = end - start + 1
            if interval_size <= max_interval_size:
                final_intervals.append((start, end))
            else:
                # Split into chunks
                num_chunks = (interval_size + max_interval_size - 1) // max_interval_size
                chunk_size = interval_size // num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start + i * chunk_size
                    chunk_end = min(start + (i + 1) * chunk_size - 1, end)
                    if chunk_end >= chunk_start:
                        final_intervals.append((chunk_start, chunk_end))
        
        return final_intervals
    
    def _run_pyod(self, X: np.ndarray, series: pd.Series = None) -> Optional[List[Tuple[int, int]]]:
        """
        Run PyOD IForest for trend change detection
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        try:
            # Use changepoint detection instead of outlier detection
            if series is not None:
                return self._detect_trend_changes(series)
            
            # Fallback to original outlier detection if series not provided
            from pyod.models.iforest import IForest
            if len(X) == 0:
                return None
            model = IForest()
            model.fit(X)
            predictions = model.predict(X)  # 1 = outlier
            # Convert to intervals
            intervals = []
            in_interval = False
            start_idx = None
            for i, pred in enumerate(predictions):
                if pred == 1 and not in_interval:
                    start_idx = i
                    in_interval = True
                elif pred == 0 and in_interval:
                    intervals.append((start_idx, i - 1))
                    in_interval = False
            if in_interval:
                intervals.append((start_idx, len(predictions) - 1))
            return intervals if intervals else None
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def _run_alibi_detect(self, X: np.ndarray, series: pd.Series = None) -> Optional[List[Tuple[int, int]]]:
        """
        Run alibi-detect for trend change detection
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        try:
            # Use changepoint detection instead of outlier detection
            if series is not None:
                return self._detect_trend_changes(series)
            
            # Fallback to original outlier detection if series not provided
            import warnings
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            from io import StringIO
            
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                warnings.filterwarnings('ignore', message='.*PyTorch.*')
                warnings.filterwarnings('ignore', message='.*TensorFlow.*')
                warnings.filterwarnings('ignore', message='.*Flax.*')
                warnings.filterwarnings('ignore', message='.*threshold.*')
                warnings.filterwarnings('ignore', message='.*infer_threshold.*')
                warnings.filterwarnings('ignore', category=UserWarning)
                
                # Suppress stdout/stderr from alibi-detect
                dummy_stdout = StringIO()
                dummy_stderr = StringIO()
                
                with redirect_stdout(dummy_stdout), redirect_stderr(dummy_stderr):
                    from alibi_detect.od import IForest as AlibiIForest
                    
                    if len(X) == 0 or X.shape[0] < 3:
                        return None
                    
                    detector = AlibiIForest(threshold=None)
                    detector.fit(X)
                    
                    try:
                        detector.infer_threshold(X, threshold_perc=95.0)
                    except Exception:
                        pass
                    
                    pred = detector.predict(X)
                
                # Convert to intervals
                if isinstance(pred, dict) and 'data' in pred:
                    if 'is_outlier' in pred['data']:
                        outliers = pred['data']['is_outlier']
                        if hasattr(outliers, 'astype'):
                            outliers = outliers.astype(int)
                        else:
                            outliers = np.array([1 if x else 0 for x in outliers], dtype=int)
                        
                        # Convert to intervals
                        intervals = []
                        in_interval = False
                        start_idx = None
                        for i, outlier in enumerate(outliers):
                            if outlier == 1 and not in_interval:
                                start_idx = i
                                in_interval = True
                            elif outlier == 0 and in_interval:
                                intervals.append((start_idx, i - 1))
                                in_interval = False
                        if in_interval:
                            intervals.append((start_idx, len(outliers) - 1))
                        return intervals if intervals else None
                    elif 'outlier_score' in pred['data']:
                        # Use score-based threshold (top 5% as outliers)
                        scores = pred['data']['outlier_score']
                        if scores is not None and len(scores) > 0:
                            threshold = np.percentile(scores, 95)
                            outliers = (scores > threshold).astype(int)
                            
                            # Convert to intervals
                            intervals = []
                            in_interval = False
                            start_idx = None
                            for i, outlier in enumerate(outliers):
                                if outlier == 1 and not in_interval:
                                    start_idx = i
                                    in_interval = True
                                elif outlier == 0 and in_interval:
                                    intervals.append((start_idx, i - 1))
                                    in_interval = False
                            if in_interval:
                                intervals.append((start_idx, len(outliers) - 1))
                            return intervals if intervals else None
                return None
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def _run_dtaianomaly(self, X: np.ndarray, series: pd.Series = None) -> Optional[List[Tuple[int, int]]]:
        """
        Run dtaianomaly for trend change detection
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        try:
            # Use changepoint detection instead of outlier detection
            if series is not None:
                return self._detect_trend_changes(series)
            
            # Fallback to original outlier detection if series not provided
            if len(X) == 0 or X.shape[0] < 3:
                return None
            if X.ndim > 1:
                # Use first column as time series (original values)
                X_1d = X[:, 0]
            else:
                X_1d = X.flatten()
            
            from dtaianomaly import IsolationForest
            model = IsolationForest()
            model.fit(X_1d)
            predictions = model.predict(X_1d)
            # Convert to binary (1 = outlier, -1 = normal)
            if isinstance(predictions, np.ndarray):
                predictions = (predictions == -1).astype(int)
            else:
                predictions_array = np.array(predictions)
                predictions = (predictions_array == -1).astype(int)
            
            # Convert to intervals
            intervals = []
            in_interval = False
            start_idx = None
            for i, pred in enumerate(predictions):
                if pred == 1 and not in_interval:
                    start_idx = i
                    in_interval = True
                elif pred == 0 and in_interval:
                    intervals.append((start_idx, i - 1))
                    in_interval = False
            if in_interval:
                intervals.append((start_idx, len(predictions) - 1))
            return intervals if intervals else None
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def _run_sklearn(self, X: np.ndarray, series: pd.Series = None) -> Optional[List[Tuple[int, int]]]:
        """
        Run sklearn IsolationForest for trend change detection
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        try:
            # Use changepoint detection instead of outlier detection
            if series is not None:
                return self._detect_trend_changes(series)
            
            # Fallback to original outlier detection if series not provided
            from sklearn.ensemble import IsolationForest
            if len(X) == 0:
                return None
            model = IsolationForest(random_state=42)
            model.fit(X)
            predictions = model.predict(X)
            # Convert to binary (1 = outlier, -1 = normal)
            predictions = (predictions == -1).astype(int)
            
            # Convert to intervals
            intervals = []
            in_interval = False
            start_idx = None
            for i, pred in enumerate(predictions):
                if pred == 1 and not in_interval:
                    start_idx = i
                    in_interval = True
                elif pred == 0 and in_interval:
                    intervals.append((start_idx, i - 1))
                    in_interval = False
            if in_interval:
                intervals.append((start_idx, len(predictions) - 1))
            return intervals if intervals else None
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def run_tool(self, tool_name: str, X: np.ndarray, series: pd.Series = None) -> Optional[List[Tuple[int, int]]]:
        """
        Run a specific tool on data for trend change detection
        
        Returns:
            List of (start_idx, end_idx) tuples representing intervals with trend changes
        """
        tool_name_lower = tool_name.lower().strip()
        
        # Normalize tool names for matching
        if 'pyod' in tool_name_lower:
            return self._run_pyod(X, series=series)
        elif 'alibi' in tool_name_lower:
            return self._run_alibi_detect(X, series=series)
        elif 'dtaianomaly' in tool_name_lower or 'dtai' in tool_name_lower:
            return self._run_dtaianomaly(X, series=series)
        elif 'sklearn' in tool_name_lower or 'scikit' in tool_name_lower or 'scikit-learn' in tool_name_lower:
            return self._run_sklearn(X, series=series)
        else:
            return None
    
    def run_research(self) -> Dict[str, Any]:
        """Run research on all contexts"""
        # Load data
        df = self.load_data()
        
        if df.empty:
            return {
                'success': False,
                'error': 'No data loaded'
            }
        
        # Get context fields and metric fields
        context_fields = self.config.get('context_fields', [])
        metric_fields = self.config.get('metric_fields', [])
        timestamp_field = self.config.get('timestamp_field', 'timestamp')
        
        if not metric_fields or len(metric_fields) < 2:
            return {
                'success': False,
                'error': 'Invalid metric_fields in config'
            }
        
        metric_name_field = metric_fields[0]
        metric_value_field = metric_fields[1]
        
        # Group by context
        contexts = {}
        
        for _, row in df.iterrows():
            context_hash = self._get_context_hash(row, context_fields)
            context_display = self._get_context_display(row, context_fields)
            
            if context_hash not in contexts:
                contexts[context_hash] = {
                    'context_hash': context_hash,
                    'context_display': context_display,
                    'metrics': {}
                }
            
            metric_name = str(row[metric_name_field])
            if metric_name not in contexts[context_hash]['metrics']:
                contexts[context_hash]['metrics'][metric_name] = {
                    'name': metric_name,
                    'timestamps': [],
                    'values': []
                }
            
            if timestamp_field in row:
                contexts[context_hash]['metrics'][metric_name]['timestamps'].append(row[timestamp_field])
            contexts[context_hash]['metrics'][metric_name]['values'].append(row[metric_value_field])
        
        # Run tools on each context and metric
        results = {
            'success': True,
            'contexts': []
        }
        
        for context_hash, context_data in contexts.items():
            context_result = {
                'context_hash': context_hash,
                'context_display': context_data['context_display'],
                'metrics': []
            }
            
            for metric_name, metric_data in context_data['metrics'].items():
                # Create time series
                timestamps = pd.to_datetime(metric_data['timestamps'])
                values = pd.Series(metric_data['values'], index=timestamps)
                values = values.sort_index()
                
                if len(values) < 3:
                    continue
                
                # Remove outliers before analysis (for trend change detection)
                values_cleaned, outlier_mask = self._remove_outliers(values, method='iqr')
                
                # Prepare data for tools (with outlier removal)
                X = self._prepare_data_for_tool(values_cleaned, remove_outliers=False)  # Already cleaned
                
                if len(X) == 0:
                    continue
                
                # Run all tools - pass cleaned series for changepoint detection
                tool_results = {}
                for tool in self.tools:
                    tool_name = tool['name']
                    intervals = self.run_tool(tool_name, X, series=values_cleaned)
                    if intervals is not None:
                        tool_results[tool_name] = intervals
                
                # Store results - intervals are already lists of tuples
                tool_results_list = {}
                for tool_name, intervals in tool_results.items():
                    tool_results_list[tool_name] = intervals  # Already list of (start_idx, end_idx) tuples
                
                metric_result = {
                    'metric_name': metric_name,
                    'timestamps': values.index.tolist(),  # Original timestamps (not cleaned)
                    'values': values.values.tolist(),  # Original values (not cleaned)
                    'tool_results': tool_results_list  # Intervals for each tool
                }
                context_result['metrics'].append(metric_result)
            
            # Generate graph for this context
            if context_result['metrics']:
                try:
                    graph_json = self._generate_graph(context_result)
                    if graph_json:
                        context_result['graph_json'] = graph_json
                except Exception as e:
                    # Silently skip graph generation if it fails
                    pass
                results['contexts'].append(context_result)
        
        return results
    
    def _generate_graph(self, context_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plotly graph for context results"""
        metrics = context_result['metrics']
        num_metrics = len(metrics)
        
        if num_metrics == 0:
            return None
        
        # Create subplots
        if num_metrics > 1:
            fig = make_subplots(
                rows=num_metrics, cols=1,
                subplot_titles=[m['metric_name'] for m in metrics],
                vertical_spacing=0.1,
                shared_xaxes=True
            )
        else:
            fig = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        tool_colors = {
            'PyOD': '#FF6B6B',
            'alibi-detect': '#4ECDC4',
            'dtaianomaly': '#45B7D1',
            'Scikit-learn': '#FFA07A',
            'sklearn': '#FFA07A',
            'scikit-learn': '#FFA07A'
        }
        
        def get_tool_color(tool_name: str) -> str:
            """Get color for tool, supporting variations in naming"""
            tool_lower = tool_name.lower().strip()
            # Check exact matches first
            if tool_name in tool_colors:
                return tool_colors[tool_name]
            # Check variations
            if 'pyod' in tool_lower:
                return tool_colors['PyOD']
            elif 'alibi' in tool_lower:
                return tool_colors['alibi-detect']
            elif 'dtaianomaly' in tool_lower or 'dtai' in tool_lower:
                return tool_colors['dtaianomaly']
            elif 'sklearn' in tool_lower or 'scikit' in tool_lower:
                return tool_colors['Scikit-learn']
            # Default color
            return colors[len(metric['tool_results']) % len(colors)]
        
        for metric_idx, metric in enumerate(metrics):
            if num_metrics > 1:
                row = metric_idx + 1
                col = 1
            else:
                row = None
                col = None
            
            timestamps = pd.to_datetime(metric['timestamps'])
            values = metric['values']
            
            # Convert timestamps to strings for JSON serialization
            timestamps_str = [pd.Timestamp(ts).isoformat() if not isinstance(ts, str) else ts for ts in timestamps]
            
            # Plot original data
            if row is not None:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps_str,
                        y=values,
                        mode='lines+markers',
                        name=f"{metric['metric_name']} (data)",
                        line=dict(color='#333', width=2),
                        marker=dict(size=4),
                        showlegend=(metric_idx == 0)
                    ),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps_str,
                        y=values,
                        mode='lines+markers',
                        name=f"{metric['metric_name']} (data)",
                        line=dict(color='#333', width=2),
                        marker=dict(size=4),
                        showlegend=(metric_idx == 0)
                    )
                )
            
            # Plot tool results - intervals of trend changes
            tool_idx = 0
            for tool_name, intervals in metric['tool_results'].items():
                # intervals is now a list of (start_idx, end_idx) tuples
                if not isinstance(intervals, list) or len(intervals) == 0:
                    continue
                
                # Check if intervals are tuples (new format) or array (old format)
                if len(intervals) > 0 and isinstance(intervals[0], tuple):
                    # New format: list of (start_idx, end_idx) tuples
                    # Color will be determined by trend direction for each interval
                    
                    # Add shaded rectangles for each interval
                    for interval_idx, (start_idx, end_idx) in enumerate(intervals):
                        # Ensure indices are within bounds
                        start_idx = max(0, min(start_idx, len(timestamps) - 1))
                        end_idx = max(0, min(end_idx, len(timestamps) - 1))
                        
                        if start_idx >= len(timestamps) or end_idx >= len(timestamps) or start_idx > end_idx:
                            continue
                        
                        # Get timestamps and values for this interval
                        interval_timestamps = timestamps[start_idx:end_idx + 1]
                        interval_values = values[start_idx:end_idx + 1]
                        
                        if len(interval_timestamps) == 0:
                            continue
                        
                        # Determine trend direction in interval
                        # Compare values at start and end of interval
                        start_value = interval_values[0] if len(interval_values) > 0 else None
                        end_value = interval_values[-1] if len(interval_values) > 0 else None
                        
                        # If we have enough points, use average of first and last 20% for more robust detection
                        if len(interval_values) >= 5:
                            window_size = max(1, len(interval_values) // 5)
                            start_avg = np.mean(interval_values[:window_size])
                            end_avg = np.mean(interval_values[-window_size:])
                            trend_direction = 'positive' if end_avg > start_avg else 'negative'
                        elif start_value is not None and end_value is not None:
                            trend_direction = 'positive' if end_value > start_value else 'negative'
                        else:
                            trend_direction = 'neutral'
                        
                        # Choose color based on trend direction
                        if trend_direction == 'positive':
                            # Green for positive trend (growth)
                            tool_color = '#4CAF50'  # Green
                        elif trend_direction == 'negative':
                            # Red for negative trend (decline)
                            tool_color = '#F44336'  # Red
                        else:
                            # Gray for neutral/no clear trend
                            tool_color = '#9E9E9E'  # Gray
                        
                        # Convert timestamps to strings
                        interval_timestamps_str = [pd.Timestamp(ts).isoformat() if not isinstance(ts, str) else ts for ts in interval_timestamps]
                        
                        # Only show name for first interval of each tool
                        is_first_interval = (interval_idx == 0)
                        trend_label = "рост" if trend_direction == 'positive' else "падение" if trend_direction == 'negative' else "нейтральный"
                        interval_name = f"{tool_name} ({trend_label})" if is_first_interval else None
                        change_points_name = f"{tool_name} (change points)" if is_first_interval else None
                        
                        # Create shaded rectangle for interval
                        # Use min/max from all values for consistent visualization
                        all_values_min = min(values)
                        all_values_max = max(values)
                        range_height = all_values_max - all_values_min
                        padding = range_height * 0.05  # 5% padding
                        
                        min_y = all_values_min - padding
                        max_y = all_values_max + padding
                        
                        # Create rectangle coordinates - full height rectangle for better visibility
                        rect_x = [interval_timestamps_str[0], interval_timestamps_str[0], 
                                 interval_timestamps_str[-1], interval_timestamps_str[-1], 
                                 interval_timestamps_str[0]]
                        rect_y = [min_y, max_y, max_y, min_y, min_y]
                        
                        if row is not None:
                            # Add filled area (rectangle) for interval
                            fig.add_trace(
                                go.Scatter(
                                    x=rect_x,
                                    y=rect_y,
                                    mode='lines',
                                    name=interval_name,
                                    fill='toself',
                                    fillcolor=tool_color,
                                    opacity=0.15,
                                    line=dict(width=0),
                                    showlegend=(metric_idx == 0 and is_first_interval),
                                    legendgroup=tool_name,
                                    hoverinfo='skip'
                                ),
                                row=row, col=col
                            )
                            
                            # Add vertical dashed lines at start and end of interval (separate lines, not diagonal)
                            # Start line
                            fig.add_trace(
                                go.Scatter(
                                    x=[interval_timestamps_str[0], interval_timestamps_str[0]],
                                    y=[min_y, max_y],
                                    mode='lines+markers',
                                    name=change_points_name if is_first_interval else None,
                                    line=dict(color=tool_color, width=2, dash='dash'),
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='triangle-down',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=(metric_idx == 0 and is_first_interval),
                                    legendgroup=tool_name
                                ),
                                row=row, col=col
                            )
                            # End line
                            fig.add_trace(
                                go.Scatter(
                                    x=[interval_timestamps_str[-1], interval_timestamps_str[-1]],
                                    y=[min_y, max_y],
                                    mode='lines+markers',
                                    name=None,  # Don't duplicate legend entry
                                    line=dict(color=tool_color, width=2, dash='dash'),
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='triangle-up',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=False,
                                    legendgroup=tool_name
                                ),
                                row=row, col=col
                            )
                        else:
                            # Add filled area (rectangle) for interval
                            fig.add_trace(
                                go.Scatter(
                                    x=rect_x,
                                    y=rect_y,
                                    mode='lines',
                                    name=interval_name,
                                    fill='toself',
                                    fillcolor=tool_color,
                                    opacity=0.15,
                                    line=dict(width=0),
                                    showlegend=(metric_idx == 0 and is_first_interval),
                                    legendgroup=tool_name,
                                    hoverinfo='skip'
                                )
                            )
                            
                            # Add vertical dashed lines at start and end of interval (separate lines, not diagonal)
                            # Start line
                            fig.add_trace(
                                go.Scatter(
                                    x=[interval_timestamps_str[0], interval_timestamps_str[0]],
                                    y=[min_y, max_y],
                                    mode='lines+markers',
                                    name=change_points_name if is_first_interval else None,
                                    line=dict(color=tool_color, width=2, dash='dash'),
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='triangle-down',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=(metric_idx == 0 and is_first_interval),
                                    legendgroup=tool_name
                                )
                            )
                            # End line
                            fig.add_trace(
                                go.Scatter(
                                    x=[interval_timestamps_str[-1], interval_timestamps_str[-1]],
                                    y=[min_y, max_y],
                                    mode='lines+markers',
                                    name=None,  # Don't duplicate legend entry
                                    line=dict(color=tool_color, width=2, dash='dash'),
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='triangle-up',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=False,
                                    legendgroup=tool_name
                                )
                            )
                else:
                    # Old format: array of 0/1 (fallback for compatibility)
                    if len(intervals) != len(values):
                        continue
                    
                    pred_array = np.array(intervals)
                    outlier_indices = np.where(pred_array == 1)[0]
                    if len(outlier_indices) > 0:
                        outlier_timestamps = [timestamps[i] for i in outlier_indices]
                        outlier_values = [values[i] for i in outlier_indices]
                        
                        tool_color = get_tool_color(tool_name)
                        outlier_timestamps_str = [pd.Timestamp(ts).isoformat() if not isinstance(ts, str) else ts for ts in outlier_timestamps]
                        
                        if row is not None:
                            fig.add_trace(
                                go.Scatter(
                                    x=outlier_timestamps_str,
                                    y=outlier_values,
                                    mode='markers',
                                    name=f"{tool_name} (anomalies)",
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='x',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=(metric_idx == 0)
                                ),
                                row=row, col=col
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=outlier_timestamps_str,
                                    y=outlier_values,
                                    mode='markers',
                                    name=f"{tool_name} (anomalies)",
                                    marker=dict(
                                        color=tool_color,
                                        size=10,
                                        symbol='x',
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=(metric_idx == 0)
                                )
                            )
            
            # Update y-axis title
            if row is not None:
                fig.update_yaxes(title_text=metric['metric_name'], row=row, col=col)
        
        # Update layout
        fig.update_layout(
            height=400 * num_metrics,
            title=f"Fast Research Results: {context_result['context_display']}",
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if num_metrics > 1:
            fig.update_xaxes(title_text="Time", row=num_metrics, col=1)
        else:
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Value")
        
        # Convert figure to dict for JSON serialization
        # Plotly figures are complex objects, use to_dict() method
        fig_dict = fig.to_dict()
        
        # Return as dict with data and layout
        return {
            'data': fig_dict['data'],
            'layout': fig_dict['layout']
        }

