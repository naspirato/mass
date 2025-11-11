#!/usr/bin/env python3

"""
Visualization module for analytics events
Generates plotly interactive charts for detected events
"""

import os
import pandas as pd
from typing import Dict, Any, List


class EventVisualizer:
    """Generate visualizations for detected events"""
    
    @staticmethod
    def generate_visualizations(visualization_data: Dict[str, List[Dict[str, Any]]], 
                                output_dir: str, timestamp: str, job_name: str):
        """
        Generate plotly visualizations for contexts with events
        
        Args:
            visualization_data: Dict mapping context_hash to list of dicts with series, baseline_result, events, context info
            output_dir: Directory to save visualizations
            timestamp: Timestamp string for filenames
            job_name: Job name for filenames
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("  ⚠ plotly not available, skipping visualizations")
            return
        
        print(f"\n  Generating visualizations for {len(visualization_data)} contexts...")
        
        files_generated = 0
        for idx, (context_hash, metrics_data) in enumerate(visualization_data.items()):
            try:
                if not metrics_data:
                    print(f"    ⚠ Skipping visualization for context {idx}: no metrics data")
                    continue
                
                # Get context info from first metric (all should have same context_hash and context_json)
                first_metric = metrics_data[0]
                context_json = first_metric.get('context_json', '{}')
                safe_context_hash = context_hash[:8]  # Use first 8 chars for filename
                
                # Generate separate visualization file for each metric
                for metric_data in metrics_data:
                    series = metric_data.get('series')
                    baseline_result = metric_data.get('baseline_result', {})
                    events = metric_data.get('events', [])
                    metric_name = metric_data.get('metric_name', 'Unknown')
                    
                    # Check if series is empty
                    if series is None or series.empty:
                        continue
                    
                    # Create a single figure for this metric (no subplots)
                    fig = go.Figure()
                    
                    # Get baseline series if available
                    baseline_series = baseline_result.get('baseline_series')
                    upper_threshold = baseline_result.get('upper_threshold')
                    lower_threshold = baseline_result.get('lower_threshold')
                    
                    # Plot metric values - handle single point case
                    if len(series) > 0:
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode='lines+markers' if len(series) > 1 else 'markers',
                            name=f'{metric_name} Value',
                            line=dict(color='blue', width=2) if len(series) > 1 else None,
                            marker=dict(size=8 if len(series) == 1 else 4)
                        ))
                    
                    # Plot baseline if available
                    if baseline_series is not None and not baseline_series.empty:
                        fig.add_trace(go.Scatter(
                            x=baseline_series.index,
                            y=baseline_series.values,
                            mode='lines',
                            name=f'{metric_name} Baseline',
                            line=dict(color='green', width=2, dash='dash')
                        ))
                    
                    # Plot thresholds
                    if upper_threshold is not None:
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=[upper_threshold] * len(series),
                            mode='lines',
                            name=f'{metric_name} Upper Threshold',
                            line=dict(color='red', width=1, dash='dot')
                        ))
                    
                    if lower_threshold is not None:
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=[lower_threshold] * len(series),
                            mode='lines',
                            name=f'{metric_name} Lower Threshold',
                            line=dict(color='orange', width=1, dash='dot')
                        ))
                    
                    # Highlight event periods for this metric
                    for event in events:
                        event_start = pd.to_datetime(event['event_start_time'])
                        event_end = pd.to_datetime(event.get('event_end_time', event_start))
                        event_type = event['event_type']
                        
                        # Color based on event type
                        if 'degradation' in event_type:
                            color = 'rgba(255, 0, 0, 0.2)'
                        elif 'improvement' in event_type:
                            color = 'rgba(0, 255, 0, 0.2)'
                        else:
                            color = 'rgba(255, 255, 0, 0.2)'
                        
                        # Add shaded region for event period
                        fig.add_vrect(
                            x0=event_start,
                            x1=event_end,
                            fillcolor=color,
                            layer="below",
                            line_width=0,
                            annotation_text=f"{metric_name}: {event_type}",
                            annotation_position="top left"
                        )
                        
                        # Add marker at event start
                        event_value = series.get(event_start, None)
                        if event_value is None:
                            # Find closest point
                            closest_idx = series.index.get_indexer([event_start], method='nearest')[0]
                            if closest_idx >= 0:
                                event_value = series.iloc[closest_idx]
                                event_time = series.index[closest_idx]
                            else:
                                continue
                        else:
                            event_time = event_start
                        
                        fig.add_trace(go.Scatter(
                            x=[event_time],
                            y=[event_value],
                            mode='markers',
                            name=f"{metric_name}: {event_type}",
                            marker=dict(size=12, symbol='star', color='red' if 'degradation' in event_type else 'green'),
                            showlegend=False
                        ))
                    
                    # Add annotations about what couldn't be determined
                    annotations = []
                    
                    if not events or len(events) == 0:
                        annotations.append(dict(
                            xref="paper", yref="paper",
                            x=0.5, y=0.98,
                            text="⚠ События не обнаружены",
                            showarrow=False,
                            font=dict(size=14, color="orange"),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="orange",
                            borderwidth=1
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{metric_name} - Events Visualization<br><sub>{context_json[:100]}...</sub>",
                        xaxis_title="Time",
                        yaxis_title=metric_name,
                        hovermode='x unified',
                        height=400,
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        annotations=annotations
                    )
                    
                    # Save to HTML file - one file per metric
                    # Sanitize metric name for filename
                    safe_metric_name = metric_name.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
                    filename = f"{job_name}_event_{safe_context_hash}_{safe_metric_name}_{timestamp}.html"
                    filepath = os.path.join(output_dir, filename)
                    fig.write_html(filepath)
                    
                    # Save data for this visualization (for debugging and test creation)
                    # Save series, baseline_result, events, and context info
                    data_filename = filename.replace('.html', '_data.json')
                    data_filepath = os.path.join(output_dir, data_filename)
                    
                    # Prepare data for JSON serialization
                    # Convert series to dict with index and values
                    series_dict = {
                        'index': [str(idx) for idx in series.index],
                        'values': series.values.tolist()
                    }
                    
                    # Convert baseline_series if available
                    baseline_series_dict = None
                    if baseline_series is not None and not baseline_series.empty:
                        baseline_series_dict = {
                            'index': [str(idx) for idx in baseline_series.index],
                            'values': baseline_series.values.tolist()
                        }
                    
                    # Prepare baseline_result (remove non-serializable items)
                    baseline_result_serializable = {
                        'baseline_value': float(baseline_result.get('baseline_value', 0)) if baseline_result.get('baseline_value') is not None else None,
                        'upper_threshold': float(baseline_result.get('upper_threshold', 0)) if baseline_result.get('upper_threshold') is not None else None,
                        'lower_threshold': float(baseline_result.get('lower_threshold', 0)) if baseline_result.get('lower_threshold') is not None else None,
                        'baseline_method': baseline_result.get('baseline_method'),
                        'window_size': baseline_result.get('window_size'),
                        'sensitivity': float(baseline_result.get('sensitivity', 0)) if baseline_result.get('sensitivity') is not None else None,
                        'adaptive_threshold': baseline_result.get('adaptive_threshold', False)
                    }
                    
                    # Prepare events (convert timestamps to strings)
                    events_serializable = []
                    for event in events:
                        event_serializable = {}
                        for key, value in event.items():
                            if hasattr(value, 'strftime'):  # datetime objects
                                event_serializable[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                                event_serializable[key] = str(value)
                            elif hasattr(value, 'item'):  # numpy types
                                try:
                                    event_serializable[key] = value.item()
                                except (AttributeError, ValueError):
                                    event_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
                            else:
                                event_serializable[key] = value
                        events_serializable.append(event_serializable)
                    
                    # Save all data including variant config
                    import json
                    variant_config = metric_data.get('variant_config', {})
                    saved_data = {
                        'metric_name': metric_name,
                        'context_hash': context_hash,
                        'context_json': context_json,
                        'series': series_dict,
                        'baseline_series': baseline_series_dict,
                        'baseline_result': baseline_result_serializable,
                        'events': events_serializable,
                        'variant_config': variant_config,
                        'timestamp': timestamp,
                        'job_name': job_name
                    }
                    
                    with open(data_filepath, 'w', encoding='utf-8') as f:
                        json.dump(saved_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    files_generated += 1
                
            except Exception as e:
                print(f"    ⚠ Could not generate visualization for context {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  ✓ Generated {files_generated} visualization files in {output_dir}")

