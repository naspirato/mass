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
            
            return self.data_access.load_data_from_file(str(data_file))
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
    
    def _prepare_data_for_tool(self, series: pd.Series) -> np.ndarray:
        """Prepare time series data for anomaly detection tools"""
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
    
    def _run_pyod(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Run PyOD IForest"""
        try:
            from pyod.models.iforest import IForest
            if len(X) == 0:
                return None
            model = IForest()
            model.fit(X)
            predictions = model.predict(X)  # 1 = outlier
            return predictions
        except ImportError:
            return None
        except Exception as e:
            print(f"PyOD error: {e}")
            return None
    
    def _run_alibi_detect(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Run alibi-detect"""
        try:
            # Suppress warnings about missing ML libraries (must be before import)
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
                
                # Suppress stdout/stderr from alibi-detect (it prints messages like "No threshold level set")
                # Create dummy output streams
                dummy_stdout = StringIO()
                dummy_stderr = StringIO()
                
                with redirect_stdout(dummy_stdout), redirect_stderr(dummy_stderr):
                    from alibi_detect.od import IForest as AlibiIForest
                    
                    if len(X) == 0 or X.shape[0] < 3:
                        return None
                    
                    # Simple usage - need to infer threshold first
                    # All output during detector creation and inference will be suppressed
                    detector = AlibiIForest(threshold=None)
                    detector.fit(X)
                    
                    # Infer threshold from training data
                    try:
                        detector.infer_threshold(X, threshold_perc=95.0)
                    except Exception:
                        # If infer_threshold fails, use default approach
                        # Try to set a threshold manually based on scores
                        try:
                            pred_temp = detector.predict(X)
                            if isinstance(pred_temp, dict) and 'data' in pred_temp:
                                if 'outlier_score' in pred_temp['data']:
                                    scores = pred_temp['data']['outlier_score']
                                    if scores is not None and len(scores) > 0:
                                        threshold = np.percentile(scores, 95)
                                        detector.threshold = threshold
                        except Exception:
                            pass
                    
                    pred = detector.predict(X)
                
                # Process results outside the redirect context
                # pred is a dictionary with 'data' key containing 'is_outlier'
                if isinstance(pred, dict) and 'data' in pred:
                    if 'is_outlier' in pred['data']:
                        outliers = pred['data']['is_outlier']
                        # Handle different types
                        if hasattr(outliers, 'astype'):
                            return outliers.astype(int)
                        else:
                            return np.array([1 if x else 0 for x in outliers], dtype=int)
                    elif 'outlier_score' in pred['data']:
                        # Use score-based threshold (top 5% as outliers)
                        scores = pred['data']['outlier_score']
                        if scores is not None and len(scores) > 0:
                            threshold = np.percentile(scores, 95)
                            return (scores > threshold).astype(int)
                return None
        except ImportError:
            return None
        except Exception as e:
            # Only print error if it's not about missing ML libraries
            if 'PyTorch' not in str(e) and 'TensorFlow' not in str(e) and 'Flax' not in str(e):
                print(f"alibi-detect error: {e}")
            return None
    
    def _run_dtaianomaly(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Run dtaianomaly"""
        try:
            # dtaianomaly is for time series, need 1D array
            if len(X) == 0 or X.shape[0] < 3:
                return None
            if X.ndim > 1:
                # Use first column as time series (original values)
                X_1d = X[:, 0]
            else:
                X_1d = X.flatten()
            
            # dtaianomaly requires pandas Series or numpy array
            from dtaianomaly import IsolationForest
            model = IsolationForest()
            model.fit(X_1d)
            predictions = model.predict(X_1d)
            # Convert to binary (1 = outlier, -1 = normal in sklearn convention)
            if isinstance(predictions, np.ndarray):
                return (predictions == -1).astype(int)
            else:
                # Handle different return types
                predictions_array = np.array(predictions)
                return (predictions_array == -1).astype(int)
        except ImportError:
            return None
        except Exception as e:
            print(f"dtaianomaly error: {e}")
            return None
    
    def _run_sklearn(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Run sklearn IsolationForest"""
        try:
            from sklearn.ensemble import IsolationForest
            if len(X) == 0:
                return None
            model = IsolationForest(random_state=42)
            model.fit(X)
            predictions = model.predict(X)
            # Convert to binary (1 = outlier, -1 = normal)
            return (predictions == -1).astype(int)
        except ImportError:
            return None
        except Exception as e:
            print(f"sklearn error: {e}")
            return None
    
    def run_tool(self, tool_name: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Run a specific tool on data"""
        tool_name_lower = tool_name.lower().strip()
        
        # Normalize tool names for matching
        if 'pyod' in tool_name_lower:
            return self._run_pyod(X)
        elif 'alibi' in tool_name_lower:
            return self._run_alibi_detect(X)
        elif 'dtaianomaly' in tool_name_lower or 'dtai' in tool_name_lower:
            return self._run_dtaianomaly(X)
        elif 'sklearn' in tool_name_lower or 'scikit' in tool_name_lower or 'scikit-learn' in tool_name_lower:
            return self._run_sklearn(X)
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
                
                # Prepare data for tools
                X = self._prepare_data_for_tool(values)
                
                if len(X) == 0:
                    continue
                
                # Run all tools
                tool_results = {}
                for tool in self.tools:
                    tool_name = tool['name']
                    predictions = self.run_tool(tool_name, X)
                    if predictions is not None:
                        tool_results[tool_name] = predictions
                
                # Store results - convert numpy arrays to lists
                tool_results_list = {}
                for tool_name, predictions in tool_results.items():
                    if isinstance(predictions, np.ndarray):
                        tool_results_list[tool_name] = predictions.tolist()
                    else:
                        tool_results_list[tool_name] = list(predictions) if hasattr(predictions, '__iter__') else predictions
                
                metric_result = {
                    'metric_name': metric_name,
                    'timestamps': values.index.tolist(),
                    'values': values.values.tolist(),
                    'tool_results': tool_results_list
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
            
            # Plot tool results
            tool_idx = 0
            for tool_name, predictions in metric['tool_results'].items():
                if len(predictions) != len(values):
                    continue
                
                # Convert predictions to numpy array if needed
                pred_array = np.array(predictions)
                
                # Find outliers
                outlier_indices = np.where(pred_array == 1)[0]
                if len(outlier_indices) > 0:
                    outlier_timestamps = [timestamps[i] for i in outlier_indices]
                    outlier_values = [values[i] for i in outlier_indices]
                    
                    tool_color = get_tool_color(tool_name)
                    tool_idx += 1
                    
                    # Convert timestamps to strings for JSON serialization
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

