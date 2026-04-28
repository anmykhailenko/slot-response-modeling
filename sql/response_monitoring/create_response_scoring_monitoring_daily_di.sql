set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di (
    monitor_run_ts timestamp comment 'UTC timestamp when the response daily monitoring run executed',
    run_label string comment 'operator-visible run identifier for the monitoring execution',
    metric_scope string comment 'global or segment row scope for daily monitoring persistence',
    segment_column string comment 'segment column name for segment-scoped rows such as vip_level',
    segment_value string comment 'segment value for segment-scoped rows',
    monitor_status string comment 'overall daily monitoring rollup status: ok, warning, or critical',
    scoring_table string comment 'fully qualified scored response table that was monitored',
    reference_pt string comment 'reference scoring partition used for drift comparison when provided',
    partition_present_flag bigint comment '1 if the scored partition existed and returned rows during monitoring',
    write_success_flag bigint comment '1 when the monitoring script completed and prepared a write payload',
    row_count bigint comment 'scored response row count for pt',
    distinct_player_count bigint comment 'distinct player_id count in the scored partition',
    duplicate_player_count bigint comment 'duplicate player_id row count in the scored partition',
    duplicate_partition_count bigint comment 'duplicate player_id plus pt row count in the scored partition',
    invalid_score_count bigint comment 'null or out-of-range predicted_response_score row count',
    metadata_missing_rows bigint comment 'rows missing required score metadata fields',
    model_name string comment 'single model_name detected in the scored partition when stable',
    model_version string comment 'single model_version detected in the scored partition when stable',
    model_reference_path string comment 'single model_reference_path detected in the scored partition when stable',
    selected_threshold double comment 'single selected_threshold detected in the scored partition when stable',
    snapshot_date_min date comment 'minimum snapshot_date observed in the scored partition',
    snapshot_date_max date comment 'maximum snapshot_date observed in the scored partition',
    snapshot_date_distinct_count bigint comment 'number of distinct snapshot_date values observed in the scored partition',
    snapshot_date_lag_days bigint comment 'difference in days between score_date and max snapshot_date',
    score_min double comment 'minimum predicted_response_score',
    score_max double comment 'maximum predicted_response_score',
    score_mean double comment 'mean predicted_response_score',
    score_median double comment 'median predicted_response_score',
    score_std double comment 'standard deviation of predicted_response_score',
    p01 double comment '1st percentile predicted_response_score',
    p05 double comment '5th percentile predicted_response_score',
    p10 double comment '10th percentile predicted_response_score',
    p25 double comment '25th percentile predicted_response_score',
    p50 double comment '50th percentile predicted_response_score',
    p75 double comment '75th percentile predicted_response_score',
    p90 double comment '90th percentile predicted_response_score',
    p95 double comment '95th percentile predicted_response_score',
    p99 double comment '99th percentile predicted_response_score',
    bucket_distribution_json string comment 'serialized bucket-level score distribution summary for the monitored partition',
    vip_distribution_json string comment 'serialized vip_level distribution and average-score summary',
    score_by_segment_json string comment 'serialized response_priority_bucket by segment summary',
    eligible_population_json string comment 'serialized eligible population comparison summary from the feature table when available',
    alert_count bigint comment 'alert row count emitted by the daily monitoring run',
    alerts_json string comment 'serialized daily alert payload for the monitored partition'
)
comment 'Daily operational and score-behavior monitoring for the Phase-1 observational response scoring pipeline. This is operational monitoring only and not causal uplift validation.'
partitioned by (
    pt string comment 'scoring partition in yyyymmdd form'
)
lifecycle 180;
