# Analytics Configuration Guide

This directory contains YAML configuration files for the YDB Metrics Analytics System.

## Configuration Structure

Each configuration file defines:

1. **Job metadata** (`job`): Name and description of the analytics job
2. **Data source** (`data_source`): SQL query to load measurements from YDB
3. **Field mappings** (`context_fields`, `metric_fields`, `timestamp_field`): How to interpret the data
4. **Analytics parameters** (`analytics`): Baseline calculation method and parameters
5. **Event detection** (`events`): Which events to detect and filtering criteria
6. **Thresholds** (`thresholds`): Whether to keep historical threshold values
7. **Output settings** (`output`): Where and how to save results
8. **Runtime settings** (`runtime`): Timezone and timeout configuration

## Field Definitions

### context_fields
List of column names that define the context/slice for grouping metrics. For example:
- `["cluster_monitoring", "git_branch", "run_type"]` groups metrics by cluster, branch, and run type

### metric_fields
List of two field names:
- First field: metric name (e.g., "metric_name")
- Second field: metric value (e.g., "metric_value")

### timestamp_field
Column name containing the timestamp for time series analysis

## Baseline Methods

- **rolling_mean**: Simple rolling average (fast, good for stable metrics)
- **zscore**: Z-score based baseline (good for normally distributed data)
- **median**: Rolling median (robust to outliers)
- **prophet**: Facebook Prophet (handles trends and seasonality, slower)
- **adtk-levelshift**: ADTK LevelShiftDetector (detects level shifts, requires adtk)

## Event Types

- **degradation_start**: Metric value drops below lower threshold
- **improvement_start**: Metric value rises above upper threshold
- **threshold_shift**: Baseline level shifts (requires adtk-levelshift or prophet)

## Example Queries

### Single Metric
```sql
SELECT 
  run_start_datetime as timestamp,
  cluster_monitoring,
  git_branch,
  run_type,
  tpmC as metric_value,
  'tpmC' as metric_name
FROM `perfomance/tpcc`
WHERE timestamp >= Datetime("2024-08-01T00:01:00Z")
```

### Multiple Metrics
```sql
SELECT 
  run_start_datetime as timestamp,
  cluster_monitoring,
  git_branch,
  run_type,
  metric_value,
  metric_name
FROM `perfomance/metrics`
WHERE timestamp >= Datetime("2024-08-01T00:01:00Z")
  AND metric_name IN ('tpmC', 'efficiency', 'newOrder90p')
```

## Environment Variables

You can use environment variables in config files:
- `${VAR_NAME}` or `$VAR_NAME` syntax
- Example: `WHERE timestamp >= Datetime("${START_DATE}")`

## Best Practices

1. **Start with simple methods**: Use `rolling_mean` or `median` first, then try advanced methods if needed
2. **Adjust sensitivity**: Start with 2.0, increase for fewer alerts, decrease for more sensitivity
3. **Set appropriate window_size**: 7-14 days is usually good for daily data
4. **Use dry_run**: Test configurations with `dry_run: true` before writing to YDB
5. **Filter data in query**: Use WHERE clauses to limit data volume and improve performance

