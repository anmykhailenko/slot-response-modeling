set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- DataWorks partition parameter rules for this daily job:
--   assignment-side pt filters must use expression parameters such as $[yyyymmdd-1]
--   date/timestamp functions must keep string parameters such as '$[yyyymmdd-1]'
--
-- Real ODPS dependencies:
--   base population: pai_rec_prod.ads_bp_user_slot_churn_features_v2_di
--   voucher source:  SuperEngineProject.dwd_mms_user_claimed_info_di
--   sms source:      superengineproject.ods_msg_sms_send_record
--   outcome source:  SuperEngineProject.ads_bet_site_order_sum_di
--   target table:    pai_rec_prod.alg_uplift_phase1_response_dataset_di

with base_population as (
    select
        cast(player_id as string) as player_id,
        cast(concat(substr(pt, 1, 4), '-', substr(pt, 5, 2), '-', substr(pt, 7, 2)) as date) as assignment_date,
        cast(snapshot_date as date) as feature_snapshot_date,
        cast(recent_bet_cnt_7d as bigint) as recent_bet_cnt_7d,
        cast(recent_bet_amt_7d as double) as recent_bet_amt_7d,
        cast(recent_win_amt_7d as double) as recent_win_amt_7d,
        cast(recent_ggr_amt_7d as double) as recent_ggr_amt_7d,
        cast(recent_net_loss_amt_7d as double) as recent_net_loss_amt_7d,
        cast(recent_bet_days_7d as bigint) as recent_bet_days_7d,
        cast(recency_last_bet_to_t as bigint) as recency_last_bet_to_t,
        cast(pre_bet_cnt_30d as bigint) as pre_bet_cnt_30d,
        cast(pre_bet_amt_30d as double) as pre_bet_amt_30d,
        cast(pre_win_amt_30d as double) as pre_win_amt_30d,
        cast(pre_ggr_amt_30d as double) as pre_ggr_amt_30d,
        cast(pre_net_loss_amt_30d as double) as pre_net_loss_amt_30d,
        cast(pre_bet_days_30d as bigint) as pre_bet_days_30d,
        cast(recent_dep_cnt_7d as bigint) as recent_dep_cnt_7d,
        cast(recent_dep_amt_7d as double) as recent_dep_amt_7d,
        cast(recent_withdraw_cnt_7d as bigint) as recent_withdraw_cnt_7d,
        cast(recent_withdraw_amt_7d as double) as recent_withdraw_amt_7d,
        cast(recent_net_cash_in_7d as double) as recent_net_cash_in_7d,
        cast(feature_create_time as timestamp) as feature_create_time,
        cast(data_version as string) as data_version,
        cast(vip_level as string) as vip_level,
        cast(pt as string) as pt
    from pai_rec_prod.ads_bp_user_slot_churn_features_v2_di
    where pt = $[yyyymmdd-1]
      and is_eligible = 1
      and label_maturity_d3 = 1
),
voucher_treatment as (
    select
        cast(login_name as string) as player_id,
        cast(coalesce(release_datetime, clamined_datetime) as date) as assignment_date,
        min(cast(coalesce(release_datetime, clamined_datetime) as timestamp)) as first_treatment_timestamp,
        count(*) as voucher_event_count
    from SuperEngineProject.dwd_mms_user_claimed_info_di
    where pt = $[yyyymmdd-1]
      and coalesce(release_datetime, clamined_datetime) is not null
    group by
        cast(login_name as string),
        cast(coalesce(release_datetime, clamined_datetime) as date)
),
sms_treatment as (
    select
        cast(login_name as string) as player_id,
        cast(coalesce(send_time, create_time) as date) as assignment_date,
        min(cast(coalesce(send_time, create_time) as timestamp)) as first_treatment_timestamp,
        count(*) as sms_event_count
    from superengineproject.ods_msg_sms_send_record
    where pt = $[yyyymmdd-1]
      and coalesce(send_time, create_time) is not null
      and cast(state as string) = '3'
    group by
        cast(login_name as string),
        cast(coalesce(send_time, create_time) as date)
),
treatment_rollup as (
    select
        b.player_id,
        b.assignment_date,
        case
            when v.first_treatment_timestamp is null and s.first_treatment_timestamp is null then cast(null as timestamp)
            when v.first_treatment_timestamp is null then s.first_treatment_timestamp
            when s.first_treatment_timestamp is null then v.first_treatment_timestamp
            when v.first_treatment_timestamp <= s.first_treatment_timestamp then v.first_treatment_timestamp
            else s.first_treatment_timestamp
        end as treatment_timestamp,
        case when coalesce(v.voucher_event_count, 0) + coalesce(s.sms_event_count, 0) > 0 then cast(1 as bigint) else cast(0 as bigint) end as treatment_flag,
        cast(coalesce(v.voucher_event_count, 0) + coalesce(s.sms_event_count, 0) as bigint) as raw_assignment_event_count,
        case when coalesce(v.voucher_event_count, 0) > 0 then cast(1 as bigint) else cast(0 as bigint) end as has_voucher_treatment,
        case when coalesce(s.sms_event_count, 0) > 0 then cast(1 as bigint) else cast(0 as bigint) end as has_sms_treatment
    from base_population b
    left join voucher_treatment v
      on b.player_id = v.player_id
     and b.assignment_date = v.assignment_date
    left join sms_treatment s
      on b.player_id = s.player_id
     and b.assignment_date = s.assignment_date
),
outcome_daily as (
    select
        cast(login_name as string) as player_id,
        cast(stat_date as date) as stat_date,
        cast(bet_site_id as string) as bet_site_id,
        sum(cast(bet_amount as double)) as bet_amount_1d,
        sum(cast(gross_gaming_revenue as double)) as gross_ggr_1d,
        count(*) as source_row_count_1d
    from SuperEngineProject.ads_bet_site_order_sum_di
    where pt >= $[yyyymmdd]
      and pt <= $[yyyymmdd+2]
    group by
        cast(login_name as string),
        cast(stat_date as date),
        cast(bet_site_id as string)
),
outcome_rollup as (
    select
        b.player_id,
        b.assignment_date,
        coalesce(sum(o.bet_amount_1d), 0.0) as outcome_gross_bet_3d_value,
        coalesce(sum(o.gross_ggr_1d), 0.0) as outcome_gross_ggr_3d_value,
        coalesce(sum(o.source_row_count_1d), 0) as outcome_source_row_count_3d,
        count(distinct case when o.player_id is not null then concat(o.player_id, '|', cast(o.stat_date as string), '|', coalesce(o.bet_site_id, '')) end) as outcome_distinct_source_keys_3d,
        min(o.stat_date) as first_outcome_event_date,
        max(o.stat_date) as last_outcome_event_date
    from base_population b
    left join outcome_daily o
      on o.player_id = b.player_id
     and o.stat_date > b.assignment_date
     and o.stat_date <= dateadd(b.assignment_date, 3, 'dd')
    group by
        b.player_id,
        b.assignment_date
)
insert overwrite table pai_rec_prod.alg_uplift_phase1_response_dataset_di partition (pt)
select
    b.player_id,
    b.assignment_date,
    t.treatment_timestamp,
    t.treatment_flag,
    t.raw_assignment_event_count,
    t.has_voucher_treatment,
    t.has_sms_treatment,
    b.feature_snapshot_date,
    b.recent_bet_cnt_7d,
    b.recent_bet_amt_7d,
    b.recent_win_amt_7d,
    b.recent_ggr_amt_7d,
    b.recent_net_loss_amt_7d,
    b.recent_bet_days_7d,
    b.recency_last_bet_to_t,
    b.pre_bet_cnt_30d,
    b.pre_bet_amt_30d,
    b.pre_win_amt_30d,
    b.pre_ggr_amt_30d,
    b.pre_net_loss_amt_30d,
    b.pre_bet_days_30d,
    b.recent_dep_cnt_7d,
    b.recent_dep_amt_7d,
    b.recent_withdraw_cnt_7d,
    b.recent_withdraw_amt_7d,
    b.recent_net_cash_in_7d,
    b.feature_create_time,
    b.data_version,
    b.vip_level,
    cast(o.outcome_gross_bet_3d_value as double) as outcome_gross_bet_3d_value,
    cast(o.outcome_gross_ggr_3d_value as double) as outcome_gross_ggr_3d_value,
    cast(o.outcome_source_row_count_3d as bigint) as outcome_source_row_count_3d,
    cast(o.outcome_distinct_source_keys_3d as bigint) as outcome_distinct_source_keys_3d,
    cast(o.first_outcome_event_date as date) as first_outcome_event_date,
    cast(o.last_outcome_event_date as date) as last_outcome_event_date,
    case
        when coalesce(o.outcome_gross_bet_3d_value, 0.0) > 0 then cast(1 as bigint)
        else cast(0 as bigint)
    end as response_label_positive_3d,
    b.pt as pt
from base_population b
left join treatment_rollup t
  on b.player_id = t.player_id
 and b.assignment_date = t.assignment_date
left join outcome_rollup o
  on b.player_id = o.player_id
 and b.assignment_date = o.assignment_date;
