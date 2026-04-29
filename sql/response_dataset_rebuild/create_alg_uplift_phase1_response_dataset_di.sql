set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.alg_uplift_phase1_response_dataset_di (
    player_id string comment 'login_name-style player identifier',
    assignment_date date comment 'assignment date used as the outcome anchor date',
    treatment_timestamp timestamp comment 'upstream assignment timestamp if available',
    treatment_flag bigint comment 'binary treatment indicator from the upstream base source',
    raw_assignment_event_count bigint comment 'count of raw assignment events collapsed into the user-day row',
    has_voucher_treatment bigint comment '1 if voucher source contributed to the user-day row',
    has_sms_treatment bigint comment '1 if sms source contributed to the user-day row',
    feature_snapshot_date date comment 'selected pre-assignment feature snapshot date',
    recent_bet_cnt_7d bigint comment 'feature mart column',
    recent_bet_amt_7d double comment 'feature mart column',
    recent_win_amt_7d double comment 'feature mart column',
    recent_ggr_amt_7d double comment 'feature mart column',
    recent_net_loss_amt_7d double comment 'feature mart column',
    recent_bet_days_7d bigint comment 'feature mart column',
    recency_last_bet_to_t bigint comment 'feature mart column',
    pre_bet_cnt_30d bigint comment 'feature mart column',
    pre_bet_amt_30d double comment 'feature mart column',
    pre_win_amt_30d double comment 'feature mart column',
    pre_ggr_amt_30d double comment 'feature mart column',
    pre_net_loss_amt_30d double comment 'feature mart column',
    pre_bet_days_30d bigint comment 'feature mart column',
    recent_dep_cnt_7d bigint comment 'feature mart column',
    recent_dep_amt_7d double comment 'feature mart column',
    recent_withdraw_cnt_7d bigint comment 'feature mart column',
    recent_withdraw_amt_7d double comment 'feature mart column',
    recent_net_cash_in_7d double comment 'feature mart column',
    feature_create_time timestamp comment 'raw feature mart audit column',
    data_version string comment 'raw feature mart audit column',
    vip_level string comment 'raw feature mart audit column',
    outcome_gross_bet_3d_value double comment 'summed post-assignment 3-day bet amount from the isolated outcome source',
    outcome_gross_ggr_3d_value double comment 'summed post-assignment 3-day gross gaming revenue from the isolated outcome source',
    outcome_source_row_count_3d bigint comment 'count of source rows contributing to the 3-day outcome rollup',
    outcome_distinct_source_keys_3d bigint comment 'count of distinct outcome source keys contributing to the 3-day rollup',
    first_outcome_event_date date comment 'earliest post-assignment outcome date observed in the 3-day window',
    last_outcome_event_date date comment 'latest post-assignment outcome date observed in the 3-day window',
    response_label_positive_3d bigint comment '1 if outcome_gross_bet_3d_value > 0 else 0'
)
comment 'Phase-1 response modeling dataset with post-assignment observed betting outcomes. Not a causal uplift table.'
partitioned by (
    pt string comment 'assignment_date partition in yyyymmdd form'
)
lifecycle 90;
