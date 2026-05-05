set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di (
    assignment_start_date string comment 'assignment/scoring start date in yyyymmdd form for this monitoring run',
    assignment_end_date string comment 'assignment/scoring end date in yyyymmdd form for this monitoring run',
    maturity_horizon_days bigint comment 'response window days plus maturity delay required before observational evaluation',
    status string comment 'completed, not_evaluable, or failed',
    status_reason string comment 'reason when the slice is not evaluable, for example labels_not_mature',
    row_count bigint comment 'total scored rows considered for this performance summary',
    evaluated_row_count bigint comment 'rows with mature observed labels included in the evaluation',
    positive_label_count bigint comment 'observed positive-label row count',
    predicted_positive_count bigint comment 'rows predicted positive at the persisted score threshold',
    true_positive_count bigint comment 'confusion-matrix true positive count',
    false_positive_count bigint comment 'confusion-matrix false positive count',
    false_negative_count bigint comment 'confusion-matrix false negative count',
    true_negative_count bigint comment 'confusion-matrix true negative count',
    precision double comment 'precision at the persisted score threshold',
    recall double comment 'recall at the persisted score threshold',
    f1_score double comment 'f1 score at the persisted score threshold',
    response_rate_actual double comment 'observed positive-label rate',
    response_rate_predicted double comment 'predicted positive rate at the persisted score threshold',
    avg_score double comment 'average predicted_response_score',
    avg_score_positive_label double comment 'average predicted_response_score among observed positive-label rows',
    avg_score_negative_label double comment 'average predicted_response_score among observed negative-label rows',
    total_outcome_gross_bet_3d_value double comment 'total realized gross bet value over the 3-day outcome window',
    total_outcome_gross_ggr_3d_value double comment 'total realized gross ggr value over the 3-day outcome window',
    avg_outcome_gross_bet_3d_value double comment 'average realized gross bet value over the 3-day outcome window',
    avg_outcome_gross_ggr_3d_value double comment 'average realized gross ggr value over the 3-day outcome window',
    predicted_positive_total_gross_bet_3d_value double comment 'total realized gross bet value among rows predicted positive',
    predicted_positive_total_gross_ggr_3d_value double comment 'total realized gross ggr value among rows predicted positive',
    predicted_positive_avg_gross_bet_3d_value double comment 'average realized gross bet value among rows predicted positive',
    predicted_positive_avg_gross_ggr_3d_value double comment 'average realized gross ggr value among rows predicted positive',
    threshold double comment 'persisted selected_threshold used to convert scores into positive predictions',
    model_name string comment 'single model_name detected in the scored partition when stable',
    model_version string comment 'single model_version detected in the scored partition when stable',
    created_at timestamp comment 'UTC timestamp when the performance monitoring summary row was created'
)
comment 'Delayed observational response monitoring for the Phase-1 response scoring pipeline. This tracks realized post-score behavior and calibration-like diagnostics, not causal uplift validation.'
partitioned by (
    pt string comment 'scoring partition in yyyymmdd form'
)
lifecycle 180;
