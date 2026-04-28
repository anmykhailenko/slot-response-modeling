set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Expected runtime parameter:
--   ${pt}  e.g. 20260421

select
    pt,
    count(*) as monitoring_rows,
    sum(case when metric_scope not in ('global', 'segment') then 1 else 0 end) as invalid_metric_scope_rows,
    sum(case when metric_scope = 'segment' and (segment_column is null or segment_value is null) then 1 else 0 end) as missing_segment_identity_rows,
    sum(case when monitor_run_ts is null then 1 else 0 end) as missing_run_ts_rows,
    sum(case when run_label is null then 1 else 0 end) as missing_run_label_rows,
    sum(case when row_count <= 0 then 1 else 0 end) as non_positive_row_count_rows,
    sum(case when distinct_player_count <= 0 then 1 else 0 end) as non_positive_distinct_player_rows,
    sum(case when score_min < 0.0 or score_max > 1.0 then 1 else 0 end) as invalid_score_range_rows,
    sum(case when bucket_distribution_json is null then 1 else 0 end) as missing_bucket_distribution_rows,
    sum(case when vip_distribution_json is null then 1 else 0 end) as missing_vip_distribution_rows
from pai_rec_prod.ads_uplift_phase1_response_monitoring_daily_di
where pt = '${pt}'
group by pt;
