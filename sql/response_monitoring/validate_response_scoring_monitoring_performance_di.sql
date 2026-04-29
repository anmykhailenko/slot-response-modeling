set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Expected runtime parameter:
--   ${pt}  e.g. 20260417

select
    pt,
    count(*) as performance_rows,
    sum(case when row_count < 0 then 1 else 0 end) as negative_row_count_rows,
    sum(case when evaluated_row_count < 0 then 1 else 0 end) as negative_evaluated_row_count_rows,
    sum(case when positive_label_count < 0 then 1 else 0 end) as negative_positive_label_count_rows,
    sum(case when predicted_positive_count < 0 then 1 else 0 end) as negative_predicted_positive_count_rows,
    sum(case when precision < 0.0 or precision > 1.0 then 1 else 0 end) as invalid_precision_rows,
    sum(case when recall < 0.0 or recall > 1.0 then 1 else 0 end) as invalid_recall_rows,
    sum(case when f1_score < 0.0 or f1_score > 1.0 then 1 else 0 end) as invalid_f1_rows,
    sum(case when response_rate_actual < 0.0 or response_rate_actual > 1.0 then 1 else 0 end) as invalid_actual_response_rate_rows,
    sum(case when response_rate_predicted < 0.0 or response_rate_predicted > 1.0 then 1 else 0 end) as invalid_predicted_response_rate_rows,
    sum(case when avg_score < 0.0 or avg_score > 1.0 then 1 else 0 end) as invalid_avg_score_rows,
    sum(case when threshold < 0.0 or threshold > 1.0 then 1 else 0 end) as invalid_threshold_rows,
    sum(case when model_name is null or trim(model_name) = '' then 1 else 0 end) as missing_model_name_rows,
    sum(case when model_version is null or trim(model_version) = '' then 1 else 0 end) as missing_model_version_rows
from pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di
where pt = '${pt}'
group by pt;
