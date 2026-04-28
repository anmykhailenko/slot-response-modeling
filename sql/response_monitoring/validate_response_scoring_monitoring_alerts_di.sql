set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Expected runtime parameter:
--   ${pt}  e.g. 20260421

select
    pt,
    count(*) as alert_rows,
    sum(case when severity not in ('warning', 'critical') then 1 else 0 end) as invalid_severity_rows,
    sum(case when mode not in ('daily', 'performance') then 1 else 0 end) as invalid_mode_rows,
    sum(case when check_name is null then 1 else 0 end) as missing_check_name_rows,
    sum(case when message is null then 1 else 0 end) as missing_message_rows
from pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di
where pt = '${pt}'
group by pt;
