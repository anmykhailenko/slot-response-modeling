set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- DataWorks partition parameter rules for this daily validation:
--   assignment-side pt filters must use expression parameters such as $[yyyymmdd-1]
--   date/timestamp functions must keep string parameters such as '$[yyyymmdd-1]'

with base_population as (
    select
        cast(player_id as string) as player_id,
        cast(concat(substr(pt, 1, 4), '-', substr(pt, 5, 2), '-', substr(pt, 7, 2)) as date) as assignment_date,
        cast(pt as string) as pt
    from pai_rec_prod.ads_bp_user_slot_churn_features_v2_di
    where pt = $[yyyymmdd-1]
      and is_eligible = 1
      and label_maturity_d3 = 1
),
voucher_treatment as (
    select
        cast(login_name as string) as player_id,
        cast(coalesce(release_datetime, clamined_datetime) as date) as assignment_date
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
        cast(coalesce(send_time, create_time) as date) as assignment_date
    from superengineproject.ods_msg_sms_send_record
    where pt = $[yyyymmdd-1]
      and coalesce(send_time, create_time) is not null
      and cast(state as string) = '3'
    group by
        cast(login_name as string),
        cast(coalesce(send_time, create_time) as date)
),
expected_outcome as (
    select
        b.pt,
        b.player_id,
        b.assignment_date,
        coalesce(sum(cast(o.bet_amount as double)), 0.0) as expected_outcome_gross_bet_3d_value,
        count(o.login_name) as expected_outcome_source_row_count_3d,
        case when coalesce(sum(cast(o.bet_amount as double)), 0.0) > 0 then 1 else 0 end as expected_response_label_positive_3d
    from base_population b
    left join SuperEngineProject.ads_bet_site_order_sum_di o
      on cast(o.login_name as string) = b.player_id
     and cast(o.stat_date as date) > b.assignment_date
     and cast(o.stat_date as date) <= dateadd(b.assignment_date, 3, 'dd')
     and o.pt >= $[yyyymmdd]
     and o.pt <= $[yyyymmdd+2]
    group by
        b.pt,
        b.player_id,
        b.assignment_date
),
actual as (
    select
        cast(player_id as string) as player_id,
        cast(assignment_date as date) as assignment_date,
        cast(pt as string) as pt,
        cast(treatment_flag as bigint) as treatment_flag,
        cast(outcome_source_row_count_3d as bigint) as outcome_source_row_count_3d,
        cast(outcome_gross_bet_3d_value as double) as outcome_gross_bet_3d_value,
        cast(response_label_positive_3d as bigint) as response_label_positive_3d
    from pai_rec_prod.alg_uplift_phase1_response_dataset_di
    where pt = $[yyyymmdd-1]
)
select
    a.pt,
    count(*) as base_rows,
    sum(case when v.player_id is not null or s.player_id is not null then 1 else 0 end) as treatment_rows,
    sum(case when e.expected_outcome_source_row_count_3d > 0 then 1 else 0 end) as matched_outcome_rows,
    sum(case when e.expected_outcome_gross_bet_3d_value > 0 then 1 else 0 end) as expected_positive_bet_rows,
    sum(a.response_label_positive_3d) as actual_positive_labels,
    avg(cast(a.response_label_positive_3d as double)) as actual_label_rate,
    sum(case when a.response_label_positive_3d = e.expected_response_label_positive_3d then 1 else 0 end) as matching_label_rows,
    sum(case when a.response_label_positive_3d <> e.expected_response_label_positive_3d then 1 else 0 end) as mismatching_label_rows
from actual a
join expected_outcome e
  on a.pt = e.pt
 and a.player_id = e.player_id
 and a.assignment_date = e.assignment_date
left join voucher_treatment v
  on a.player_id = v.player_id
 and a.assignment_date = v.assignment_date
left join sms_treatment s
  on a.player_id = s.player_id
 and a.assignment_date = s.assignment_date
group by a.pt
order by a.pt;
