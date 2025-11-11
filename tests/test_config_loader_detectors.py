#!/usr/bin/env python3
"""
Юнит-тесты для проверки валидации детекторов в ConfigLoader.
"""

import os
import tempfile
import textwrap
import unittest

from mass.core.config_loader import ConfigLoader, ConfigError


BASE_CONFIG = textwrap.dedent(
    """
    job:
      name: analytics_job
    data_source:
      ydb:
        query: SELECT 1;
    context_fields: [operation_type]
    metric_fields: [metric_name, metric_value]
    timestamp_field: ts
    analytics:
      baseline_method: rolling_mean
      window_size: 7
      sensitivity: 2.0
    events:
      detect: [degradation_start]
    thresholds:
      keep_history: true
    output:
      write_to_ydb: false
      log_to_console: false
      dry_run: true
    """
).strip()


def _write_config(extra_yaml: str) -> str:
    """Записать временный конфиг и вернуть путь."""
    content = BASE_CONFIG + "\n" + textwrap.dedent(extra_yaml)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestConfigLoaderDetectors(unittest.TestCase):
    def tearDown(self):
        # Cleanup temporary files created during tests
        for attr in dir(self):
            if attr.startswith("_tmp_config_"):
                path = getattr(self, attr)
                if path and os.path.exists(path):
                    os.remove(path)

    def test_valid_detectors_configuration(self):
        self._tmp_config_valid = _write_config(
            """
            events:
              detect: [degradation_start, improvement_start]
              detectors:
                baseline_threshold:
                  type: threshold
                  label: Baseline Threshold
                ruptures_binseg:
                  type: ruptures
                  enabled: false
              default_detector: baseline_threshold
              compare_detectors: [baseline_threshold, ruptures_binseg]
            """
        )
        loader = ConfigLoader(self._tmp_config_valid)
        config = loader.get_config()
        self.assertIn('detectors', config['events'])
        self.assertEqual(config['events']['default_detector'], 'baseline_threshold')

    def test_invalid_default_detector_raises(self):
        self._tmp_config_invalid_default = _write_config(
            """
            events:
              detectors:
                baseline_threshold:
                  type: threshold
              default_detector: missing_detector
            """
        )
        with self.assertRaises(ConfigError):
            ConfigLoader(self._tmp_config_invalid_default)

    def test_invalid_compare_detector_raises(self):
        self._tmp_config_invalid_compare = _write_config(
            """
            events:
              detectors:
                baseline_threshold:
                  type: threshold
              compare_detectors: [baseline_threshold, unknown_detector]
            """
        )
        with self.assertRaises(ConfigError):
            ConfigLoader(self._tmp_config_invalid_compare)


if __name__ == "__main__":
    unittest.main()
