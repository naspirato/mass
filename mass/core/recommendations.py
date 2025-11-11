#!/usr/bin/env python3
"""
Recommendations generator for analytics exploration
Provides suggestions when no events are detected
"""

from typing import Dict, Any, List
import pandas as pd


class RecommendationGenerator:
    """Generate recommendations based on analysis results"""
    
    @staticmethod
    def generate_recommendations(
        series: pd.Series,
        baseline_result: Dict[str, Any],
        events_count: int,
        current_params: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations when no events are detected
        
        Args:
            series: Time series data
            baseline_result: Baseline calculation result
            events_count: Number of detected events
            current_params: Current analytics parameters
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if events_count == 0 and not series.empty:
            # Analyze data characteristics
            data_std = series.std()
            data_mean = abs(series.mean()) if series.mean() != 0 else 1.0
            cv = data_std / data_mean if data_mean != 0 else 0
            
            baseline_value = baseline_result.get('baseline_value')
            upper_threshold = baseline_result.get('upper_threshold')
            lower_threshold = baseline_result.get('lower_threshold')
            sensitivity = current_params.get('sensitivity', 2.0)
            window_size = current_params.get('window_size', 7)
            baseline_method = current_params.get('baseline_method', 'rolling_mean')
            
            # Check if data is too stable
            if cv < 0.05:
                recommendations.append(
                    f"Данные очень стабильны (коэффициент вариации={cv:.3f}). "
                    f"Попробуйте уменьшить sensitivity до {max(1.0, sensitivity * 0.7):.1f}"
                )
            
            # Check if thresholds are too wide
            if upper_threshold is not None and lower_threshold is not None and baseline_value is not None:
                threshold_range = upper_threshold - lower_threshold
                data_range = series.max() - series.min()
                
                if threshold_range > data_range * 2 and data_range > 0:
                    recommendations.append(
                        f"Пороги слишком широкие относительно диапазона данных. "
                        f"Попробуйте уменьшить sensitivity до {max(1.0, sensitivity * 0.8):.1f} "
                        f"или window_size до {max(3, int(window_size * 0.7))}"
                    )
            
            # Check if we need more data
            if len(series) < 20:
                recommendations.append(
                    f"Мало данных ({len(series)} точек). "
                    f"Уменьшите window_size до {max(3, min(7, len(series) // 3))}"
                )
            
            # Check if baseline method might be too conservative
            if baseline_method == 'median' and len(series) < 30:
                recommendations.append(
                    f"Метод 'median' может быть слишком консервативным для малых данных. "
                    f"Попробуйте 'rolling_mean' или 'zscore'"
                )
            
            # Check if sensitivity is too high
            if sensitivity > 2.5:
                recommendations.append(
                    f"Высокая чувствительность ({sensitivity}). "
                    f"Попробуйте уменьшить до {sensitivity * 0.8:.1f}"
                )
            
            # Check if window_size is too large
            if window_size > len(series) / 2:
                recommendations.append(
                    f"Размер окна ({window_size}) слишком большой для количества данных ({len(series)}). "
                    f"Попробуйте уменьшить до {max(3, int(len(series) / 4))}"
                )
            
            # Check if data has outliers that might need different approach
            if len(series) > 10:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
                if outliers > len(series) * 0.1:  # More than 10% outliers
                    recommendations.append(
                        f"Обнаружено много выбросов ({outliers} из {len(series)}). "
                        f"Попробуйте метод 'median' вместо '{baseline_method}'"
                    )
            
            # Default recommendation if nothing specific found
            if not recommendations:
                recommendations.append(
                    f"Попробуйте уменьшить sensitivity до {max(1.0, sensitivity * 0.75):.1f} "
                    f"или window_size до {max(3, int(window_size * 0.8))}"
                )
        
        return recommendations

