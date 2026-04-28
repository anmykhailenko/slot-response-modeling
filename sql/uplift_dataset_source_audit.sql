-- Step 1 audit query for the uplift dataset foundation.
-- Replace `${pt}` with the target build partition when auditing in ODPS.

select
  pt,
  treatment_type,
  treatment_group,
  count(*) as row_count,
  count(distinct player_id) as distinct_players,
  avg(cast(treatment_flag as double)) as treatment_rate,
  avg(cast(outcome_paid_activity_3d_flag as double)) as outcome_paid_activity_3d_rate,
  avg(cast(outcome_paid_activity_7d_flag as double)) as outcome_paid_activity_7d_rate
from pai_rec_prod.alg_uplift_player_assignment_training_di
where pt = '${pt}'
group by
  pt,
  treatment_type,
  treatment_group
order by
  row_count desc,
  treatment_type,
  treatment_group;
