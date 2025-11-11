#!/usr/bin/env python3
"""
Adaptive parameter tuning system for automatic parameter selection
Supports multiple tuning strategies: adaptive, ML-based, hybrid
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DataCharacteristics:
    """Характеристики временного ряда для подбора параметров"""
    coefficient_of_variation: float  # CV = std/mean
    volatility: float  # Волатильность изменений
    trend_strength: float  # Сила тренда (0-1)
    seasonality_strength: float  # Сила сезонности (0-1)
    outlier_ratio: float  # Доля выбросов (0-1)
    data_density: float  # Плотность данных (точек/час)
    baseline_stability: float  # Стабильность baseline (0-1)
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    data_points: int


@dataclass
class TuningResult:
    """Результат подбора параметров"""
    parameters: Dict[str, Any]  # Подобранные параметры
    confidence: float  # Уверенность в параметрах (0-1)
    method: str  # Метод подбора (adaptive, ml, hybrid, etc.)
    characteristics: DataCharacteristics  # Характеристики данных
    reasoning: str  # Объяснение выбора параметров


class ParameterTunerInterface(ABC):
    """Интерфейс для различных стратегий подбора параметров"""
    
    @abstractmethod
    def analyze_characteristics(self, series: pd.Series) -> DataCharacteristics:
        """Анализ характеристик временного ряда"""
        pass
    
    @abstractmethod
    def suggest_parameters(
        self, 
        characteristics: DataCharacteristics,
        metric_name: str,
        metric_direction: str,
        context_key: Optional[str] = None
    ) -> TuningResult:
        """Подбор параметров на основе характеристик"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Название метода подбора"""
        pass


class AdaptiveParameterTuner(ParameterTunerInterface):
    """Адаптивный подбор параметров на основе характеристик данных"""
    
    def analyze_characteristics(self, series: pd.Series) -> DataCharacteristics:
        """Анализ характеристик временного ряда"""
        if series.empty or len(series) < 2:
            # Возвращаем значения по умолчанию для пустых данных
            mean_val = 0.0
            std_val = 0.0
        else:
            mean_val = float(series.mean())
            std_val = float(series.std())
        
        # Коэффициент вариации
        cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
        
        # Волатильность (стандартное отклонение изменений)
        if len(series) > 1:
            diff_series = series.diff().dropna()
            volatility = float(diff_series.std()) if not diff_series.empty else 0.0
        else:
            volatility = 0.0
        
        # Сила тренда (через линейную регрессию)
        trend_strength = self._calculate_trend_strength(series)
        
        # Сила сезонности (упрощенная оценка)
        seasonality_strength = self._calculate_seasonality(series)
        
        # Доля выбросов (используя IQR)
        outlier_ratio = self._calculate_outlier_ratio(series)
        
        # Плотность данных (точек в час)
        if len(series) > 1:
            time_span_hours = (series.index[-1] - series.index[0]).total_seconds() / 3600
            data_density = len(series) / time_span_hours if time_span_hours > 0 else 0.0
        else:
            data_density = 0.0
        
        # Стабильность baseline (через вариацию скользящего среднего)
        baseline_stability = self._calculate_baseline_stability(series)
        
        return DataCharacteristics(
            coefficient_of_variation=cv,
            volatility=volatility,
            trend_strength=trend_strength,
            seasonality_strength=seasonality_strength,
            outlier_ratio=outlier_ratio,
            data_density=data_density,
            baseline_stability=baseline_stability,
            mean_value=mean_val,
            std_value=std_val,
            min_value=float(series.min()) if not series.empty else 0.0,
            max_value=float(series.max()) if not series.empty else 0.0,
            data_points=len(series)
        )
    
    def suggest_parameters(
        self,
        characteristics: DataCharacteristics,
        metric_name: str,
        metric_direction: str,
        context_key: Optional[str] = None
    ) -> TuningResult:
        """Подбор параметров на основе характеристик"""
        
        cv = characteristics.coefficient_of_variation
        volatility = characteristics.volatility
        trend = characteristics.trend_strength
        stability = characteristics.baseline_stability
        outlier_ratio = characteristics.outlier_ratio
        mean_val = characteristics.mean_value
        
        reasoning_parts = []
        
        # Адаптивный window_size
        if stability > 0.8 and cv < 0.1:
            # Стабильные данные - меньший window
            window_size = max(5, int(7 * (1 - stability * 0.3)))
            reasoning_parts.append(f"стабильные данные (stability={stability:.2f}) → window_size={window_size}")
        elif volatility > abs(mean_val) * 0.3:
            # Высокая волатильность - больший window
            window_size = min(20, int(7 * (1 + volatility / abs(mean_val) if mean_val != 0 else 1)))
            reasoning_parts.append(f"высокая волатильность → window_size={window_size}")
        else:
            window_size = 7
            reasoning_parts.append(f"стандартный window_size={window_size}")
        
        # Адаптивный sensitivity
        if cv < 0.05:
            # Очень стабильные данные - низкий sensitivity
            sensitivity = 1.0
            reasoning_parts.append(f"очень стабильные данные (CV={cv:.3f}) → sensitivity={sensitivity}")
        elif cv > 0.5:
            # Высокая вариативность - высокий sensitivity
            sensitivity = 2.5
            reasoning_parts.append(f"высокая вариативность (CV={cv:.3f}) → sensitivity={sensitivity}")
        else:
            # Средняя вариативность
            sensitivity = 1.5 + (cv - 0.05) * 2.0
            reasoning_parts.append(f"средняя вариативность (CV={cv:.3f}) → sensitivity={sensitivity:.2f}")
        
        # Адаптивный min_relative_change
        if stability > 0.9:
            # Стабильный baseline - можно быть чувствительнее
            min_rel_change = 0.05
            reasoning_parts.append(f"стабильный baseline → min_rel_change={min_rel_change}")
        elif stability < 0.5:
            # Нестабильный baseline - быть консервативнее
            min_rel_change = 0.15
            reasoning_parts.append(f"нестабильный baseline → min_rel_change={min_rel_change}")
        else:
            min_rel_change = 0.1
            reasoning_parts.append(f"стандартный min_rel_change={min_rel_change}")
        
        # Адаптивный baseline_method
        if trend > 0.7:
            baseline_method = 'prophet'  # Есть тренд
            reasoning_parts.append(f"сильный тренд (trend={trend:.2f}) → baseline_method=prophet")
        elif outlier_ratio > 0.2:
            baseline_method = 'median'  # Много выбросов
            reasoning_parts.append(f"много выбросов (outlier_ratio={outlier_ratio:.2f}) → baseline_method=median")
        else:
            baseline_method = 'rolling_mean'  # Стандартный
            reasoning_parts.append(f"стандартный baseline_method=rolling_mean")
        
        # Адаптивный min_absolute_change (если указан в конфиге)
        min_abs_change = max(0, int(abs(mean_val) * 0.01))  # 1% от среднего значения
        
        # Hysteresis points зависит от стабильности
        if stability > 0.8:
            hysteresis_points = 2
        else:
            hysteresis_points = 3
        
        parameters = {
            'window_size': window_size,
            'sensitivity': sensitivity,
            'min_relative_change': min_rel_change,
            'min_absolute_change': min_abs_change,
            'baseline_method': baseline_method,
            'adaptive_threshold': True,
            'hysteresis_points': hysteresis_points,
        }
        
        # Уверенность зависит от качества данных
        confidence = min(1.0, stability * 0.7 + (1 - cv) * 0.3)
        
        reasoning = "; ".join(reasoning_parts)
        
        return TuningResult(
            parameters=parameters,
            confidence=confidence,
            method='adaptive',
            characteristics=characteristics,
            reasoning=reasoning
        )
    
    def get_method_name(self) -> str:
        return 'adaptive'
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Вычисление силы тренда (0-1)"""
        if len(series) < 3:
            return 0.0
        
        try:
            # Линейная регрессия для оценки тренда
            x = np.arange(len(series))
            y = series.values
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            # Нормализуем slope относительно среднего значения
            mean_val = abs(series.mean()) if series.mean() != 0 else 1.0
            normalized_slope = abs(slope) / mean_val
            
            # Сила тренда (0-1)
            trend_strength = min(1.0, normalized_slope * len(series) / 100.0)
            return float(trend_strength)
        except:
            return 0.0
    
    def _calculate_seasonality(self, series: pd.Series) -> float:
        """Вычисление силы сезонности (0-1) - упрощенная версия"""
        if len(series) < 7:
            return 0.0
        
        try:
            # Простая оценка: вариация по дням недели (если есть временная структура)
            # Для упрощения возвращаем 0, можно расширить позже
            return 0.0
        except:
            return 0.0
    
    def _calculate_outlier_ratio(self, series: pd.Series) -> float:
        """Вычисление доли выбросов (0-1)"""
        if len(series) < 4:
            return 0.0
        
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return 0.0
            
            # Выбросы: значения вне [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_ratio = outliers / len(series)
            
            return float(outlier_ratio)
        except:
            return 0.0
    
    def _calculate_baseline_stability(self, series: pd.Series, window: int = 7) -> float:
        """Вычисление стабильности baseline (0-1)"""
        if len(series) < window * 2:
            return 0.5  # Недостаточно данных
        
        try:
            # Вычисляем скользящее среднее
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            
            # Вычисляем вариацию скользящего среднего
            rolling_std = rolling_mean.std()
            rolling_mean_val = abs(rolling_mean.mean()) if rolling_mean.mean() != 0 else 1.0
            
            # Стабильность обратно пропорциональна вариации
            cv_rolling = rolling_std / rolling_mean_val
            stability = 1.0 / (1.0 + cv_rolling * 10)  # Нормализуем к [0, 1]
            
            return float(min(1.0, max(0.0, stability)))
        except:
            return 0.5


class MLParameterTuner(ParameterTunerInterface):
    """ML-подход для подбора параметров (заглушка для будущей реализации)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация ML-тюнера
        
        Args:
            model_path: Путь к обученной модели (если есть)
        """
        self.model = None
        self.is_trained = False
        self.model_path = model_path
        self.fallback_tuner = AdaptiveParameterTuner()
        
        # Можно загрузить модель, если указан путь
        if model_path:
            self._load_model(model_path)
    
    def analyze_characteristics(self, series: pd.Series) -> DataCharacteristics:
        """Используем базовый анализатор"""
        return self.fallback_tuner.analyze_characteristics(series)
    
    def suggest_parameters(
        self,
        characteristics: DataCharacteristics,
        metric_name: str,
        metric_direction: str,
        context_key: Optional[str] = None
    ) -> TuningResult:
        """Подбор параметров через ML-модель"""
        
        if not self.is_trained:
            # Fallback на адаптивный подбор, если модель не обучена
            result = self.fallback_tuner.suggest_parameters(
                characteristics, metric_name, metric_direction, context_key
            )
            result.method = 'ml_fallback'
            result.reasoning = "ML модель не обучена, используется адаптивный подбор"
            return result
        
        # TODO: Реализовать предсказание через ML-модель
        # features = self._extract_features(characteristics)
        # predicted_params = self.model.predict([features])[0]
        
        # Пока возвращаем fallback
        result = self.fallback_tuner.suggest_parameters(
            characteristics, metric_name, metric_direction, context_key
        )
        result.method = 'ml'
        result.reasoning = "ML модель (заглушка - будет реализовано позже)"
        
        return result
    
    def get_method_name(self) -> str:
        return 'ml'
    
    def _load_model(self, model_path: str):
        """Загрузка обученной модели"""
        # TODO: Реализовать загрузку модели
        pass
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Обучение модели на исторических данных
        
        Args:
            training_data: Список словарей с полями:
                - characteristics: DataCharacteristics
                - optimal_params: Dict[str, Any]
                - performance_metrics: Dict[str, float]
        """
        # TODO: Реализовать обучение модели
        # from sklearn.ensemble import RandomForestRegressor
        # X = [self._extract_features(d['characteristics']) for d in training_data]
        # y = [d['optimal_params'] for d in training_data]
        # self.model.fit(X, y)
        # self.is_trained = True
        pass


class HybridParameterTuner(ParameterTunerInterface):
    """Гибридный подход: комбинация нескольких методов"""
    
    def __init__(self, tuners: List[ParameterTunerInterface], weights: Optional[List[float]] = None):
        """
        Инициализация гибридного тюнера
        
        Args:
            tuners: Список тюнеров для комбинации
            weights: Веса для каждого тюнера (если None, равные веса)
        """
        self.tuners = tuners
        if weights is None:
            weights = [1.0 / len(tuners)] * len(tuners)
        self.weights = weights
        
        # Нормализуем веса
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def analyze_characteristics(self, series: pd.Series) -> DataCharacteristics:
        """Используем первый тюнер для анализа"""
        return self.tuners[0].analyze_characteristics(series)
    
    def suggest_parameters(
        self,
        characteristics: DataCharacteristics,
        metric_name: str,
        metric_direction: str,
        context_key: Optional[str] = None
    ) -> TuningResult:
        """Комбинирование результатов нескольких тюнеров"""
        
        results = []
        for tuner in self.tuners:
            result = tuner.suggest_parameters(
                characteristics, metric_name, metric_direction, context_key
            )
            results.append(result)
        
        # Комбинируем параметры с учетом весов
        combined_params = {}
        param_types = {
            'window_size': int,
            'sensitivity': float,
            'min_relative_change': float,
            'min_absolute_change': int,
            'hysteresis_points': int,
        }
        
        for param_name, param_type in param_types.items():
            values = []
            weights_for_param = []
            
            for i, result in enumerate(results):
                if param_name in result.parameters:
                    values.append(result.parameters[param_name])
                    # Учитываем уверенность каждого тюнера
                    weights_for_param.append(self.weights[i] * result.confidence)
            
            if values:
                # Взвешенное среднее
                if param_type == int:
                    combined_params[param_name] = int(
                        np.average(values, weights=weights_for_param)
                    )
                else:
                    combined_params[param_name] = float(
                        np.average(values, weights=weights_for_param)
                    )
        
        # Для baseline_method используем голосование
        baseline_methods = [r.parameters.get('baseline_method', 'rolling_mean') for r in results]
        combined_params['baseline_method'] = max(set(baseline_methods), key=baseline_methods.count)
        
        # adaptive_threshold - если хотя бы один предлагает True
        combined_params['adaptive_threshold'] = any(
            r.parameters.get('adaptive_threshold', False) for r in results
        )
        
        # Средняя уверенность
        avg_confidence = np.average([r.confidence for r in results], weights=self.weights)
        
        # Объединенное объяснение
        reasoning_parts = [f"{r.method}: {r.reasoning}" for r in results]
        combined_reasoning = " | ".join(reasoning_parts)
        
        return TuningResult(
            parameters=combined_params,
            confidence=avg_confidence,
            method='hybrid',
            characteristics=characteristics,
            reasoning=combined_reasoning
        )
    
    def get_method_name(self) -> str:
        return 'hybrid'


# Фабрика для создания тюнеров
def create_parameter_tuner(
    method: str = 'adaptive',
    **kwargs
) -> ParameterTunerInterface:
    """
    Фабрика для создания тюнеров параметров
    
    Args:
        method: Метод подбора ('adaptive', 'ml', 'hybrid')
        **kwargs: Дополнительные параметры для тюнера
    
    Returns:
        ParameterTunerInterface instance
    """
    if method == 'adaptive':
        return AdaptiveParameterTuner()
    elif method == 'ml':
        return MLParameterTuner(**kwargs)
    elif method == 'hybrid':
        tuners = []
        if 'tuners' in kwargs:
            tuners = kwargs['tuners']
        else:
            # По умолчанию: adaptive + ml
            tuners = [
                AdaptiveParameterTuner(),
                MLParameterTuner()
            ]
        weights = kwargs.get('weights', None)
        return HybridParameterTuner(tuners, weights)
    else:
        raise ValueError(f"Unknown tuning method: {method}")

