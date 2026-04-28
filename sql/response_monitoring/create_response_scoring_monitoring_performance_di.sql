set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di (
    monitor_run_ts timestamp comment 'UTC timestamp when delayed observational monitoring executed',
    run_label string comment 'operator-visible run identifier for the monitoring execution',
    metric_scope string comment 'global, bucket, or segment row scope',
    segment_column string comment 'segment column name for segment-scoped rows such as vip_level',
    segment_value string comment 'segment value for segment-scoped rows',
    bucket_name string comment 'response_priority_bucket for bucket-scoped rows',
    row_count bigint comment 'matched scored rows included in this observational performance slice',
    positive_count bigint comment 'observed positive-response count in this slice',
    observed_positive_rate double comment 'observed 3-day positive-response rate for this slice',
    avg_predicted_score double comment 'average predicted_response_score for this slice',
    predicted_positive_rate double comment 'share of rows above the internal score threshold used for monitoring diagnostics',
    calibration_gap double comment 'absolute difference between average predicted score and observed positive rate',
    pr_auc double comment 'precision-recall AUC when evaluable',
    roc_auc double comment 'ROC AUC when evaluable',
    log_loss double comment 'log loss when evaluable',
    brier_score double comment 'brier score when evaluable',
    score_min double comment 'minimum predicted_response_score in this slice',
    score_max double comment 'maximum predicted_response_score in this slice',
    avg_observed_gross_bet_value double comment 'average realized gross bet value over the configured observational window',
    status string comment 'evaluated or not_evaluable',
    status_reason string comment 'reason when the slice is not evaluable, for example insufficient_rows'
)
comment 'Delayed observational response monitoring for the Phase-1 response scoring pipeline. This tracks realized post-score behavior and calibration-like diagnostics, not causal uplift validation.'
partitioned by (
    pt string comment 'scoring partition in yyyymmdd form'
)
lifecycle 180;
