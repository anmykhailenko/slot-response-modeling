set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Expected runtime parameter:
--   ${pt}  e.g. 20260417

select
    pt,
    count(*) as performance_rows,
    sum(case when metric_scope not in ('global', 'bucket', 'segment') then 1 else 0 end) as invalid_scope_rows,
    sum(case when row_count <= 0 then 1 else 0 end) as non_positive_row_count_rows,
    sum(case when observed_positive_rate < 0.0 or observed_positive_rate > 1.0 then 1 else 0 end) as invalid_positive_rate_rows,
    sum(case when avg_predicted_score < 0.0 or avg_predicted_score > 1.0 then 1 else 0 end) as invalid_avg_score_rows,
    sum(case when metric_scope = 'bucket' and bucket_name is null then 1 else 0 end) as missing_bucket_name_rows,
    sum(case when metric_scope = 'segment' and (segment_column is null or segment_value is null) then 1 else 0 end) as missing_segment_rows
from pai_rec_prod.ads_uplift_phase1_response_monitoring_performance_di
where pt = '${pt}'
group by pt;
