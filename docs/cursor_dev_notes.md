## Cursor Dev Notes

### Python Environment
- Требуется Python ≥ 3.12. Для локальной разработки в среде Cursor достаточно системного `python3`.
- Установка зависимостей (используется `--user`, чтобы не создавать виртуальное окружение):
  ```bash
  cd /workspace
  python3 -m pip install --user -r requirements.txt
  ```
- Скрипты из `~/.local/bin` (например, `coverage`) автоматически не попадают в `PATH`. При необходимости добавьте:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```

### Запуск аналитики
- Основной сценарий:
  ```bash
  cd /workspace
  python3 -m mass.core.analytics_job configs/example.yaml --dry-run
  ```
- Для ограничения окна анализа можно указать `--event-deepness`, например `--event-deepness 7d`.

### Юнит- и E2E-тесты
- Быстрый прогон:
  ```bash
  cd /workspace
  PYTHONPATH=/workspace python3 -m unittest discover -s tests
  ```
- Прогон с покрытием:
  ```bash
  PYTHONPATH=/workspace /home/ubuntu/.local/bin/coverage run -m unittest discover -s tests
  /home/ubuntu/.local/bin/coverage report -m
  ```
  > PyOD/ruptures-тесты запускаются автоматически, если пакеты установлены (они перечислены в `requirements.txt`). Предупреждения `sklearn` о `__sklearn_tags__` безопасны.

### Web UI
- Запуск интерфейса:
  ```bash
  cd /workspace/mass/ui
  python3 app.py
  ```
- В блоке «⚙️ Действия с конфигом» можно задать `event_deepness` и выбрать набор детекторов (Baseline, PyOD, Ruptures).  

### Быстрые ссылки
- Основной менеджер детекторов: `mass/core/detection_manager.py`
- Тесты на конфигурацию и детекторы:
  - `tests/test_config_loader_detectors.py`
  - `tests/test_detection_manager.py`
