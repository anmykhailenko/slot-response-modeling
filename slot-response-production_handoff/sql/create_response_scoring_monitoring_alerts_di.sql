set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.ads_uplift_phase1_response_monitoring_alerts_di (
    monitor_run_ts timestamp comment 'UTC timestamp when the alert was emitted',
    run_label string comment 'operator-visible run identifier for the monitoring execution',
    mode string comment 'monitoring mode that emitted the alert: daily or performance',
    severity string comment 'alert severity: warning or critical',
    check_name string comment 'stable alert check identifier',
    metric_name string comment 'metric that breached the configured rule',
    observed_value string comment 'observed metric value serialized as text',
    threshold_value string comment 'configured threshold serialized as text',
    reference_value string comment 'optional comparison baseline serialized as text',
    message string comment 'human-readable alert explanation for operators',
    context_json string comment 'serialized structured context for debugging and routing'
)
comment 'Alert sink for Phase-1 observational response scoring monitoring. Alerts describe operational or observational score-health issues only; they do not imply causal failure.'
partitioned by (
    pt string comment 'scoring partition in yyyymmdd form'
)
lifecycle 180;
