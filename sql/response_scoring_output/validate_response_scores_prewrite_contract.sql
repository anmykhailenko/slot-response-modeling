set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameters:
--   ${direct_scored_source}  relation name bound for this run; prefer a temporary or transient ODPS view
--   ${scoring_pt}            e.g. 20260420
--
-- Purpose:
--   Mandatory gate before `insert_overwrite_alg_uplift_phase1_response_scores_di.sql`.
--   The operator must verify that every violation count below is zero before overwriting
--   the final partition. If any count is non-zero, do not publish.

with source_rows as (
    select
        cast(player_id as string) as player_id,
        cast(score_date as date) as score_date,
        cast(scoring_pt as string) as scoring_pt,
        cast(scoring_ts as timestamp) as scoring_ts,
        cast(snapshot_date as date) as snapshot_date,
        cast(predicted_response_score as double) as predicted_response_score,
        cast(score_rank as bigint) as score_rank,
        cast(score_percentile as double) as score_percentile,
        cast(response_priority_bucket as string) as response_priority_bucket,
        cast(action_recommendation as string) as action_recommendation,
        cast(vip_level as string) as vip_level,
        cast(model_name as string) as model_name,
        cast(model_version as string) as model_version,
        cast(model_reference_path as string) as model_reference_path,
        cast(selected_threshold as double) as selected_threshold
    from ${direct_scored_source}
    where cast(scoring_pt as string) = '${scoring_pt}'
),
duplicate_player_ids as (
    select
        player_id,
        count(1) as duplicate_row_count
    from source_rows
    group by player_id
    having count(1) > 1
)
select
    '${scoring_pt}' as scoring_pt,
    count(1) as total_rows,
    case when count(1) > 0 then 0 else 1 end as empty_output_violation_rows,
    count(1) - count(distinct player_id) as duplicate_player_id_rows,
    count(distinct d.player_id) as duplicate_player_id_groups,
    sum(case when player_id is null or trim(player_id) = '' then 1 else 0 end) as player_id_missing_rows,
    sum(case when predicted_response_score is null then 1 else 0 end) as score_missing_rows,
    sum(case when predicted_response_score < 0.0 or predicted_response_score > 1.0 then 1 else 0 end) as score_out_of_range_rows,
    sum(case when response_priority_bucket is null or trim(response_priority_bucket) = '' then 1 else 0 end) as bucket_missing_rows,
    sum(case when response_priority_bucket not in ('very_low', 'low', 'medium', 'high', 'very_high') then 1 else 0 end) as invalid_bucket_rows,
    sum(case when score_date is null then 1 else 0 end) as score_date_missing_rows,
    sum(case when scoring_pt is null or trim(scoring_pt) = '' then 1 else 0 end) as scoring_pt_missing_rows,
    sum(case when scoring_pt <> '${scoring_pt}' then 1 else 0 end) as scoring_pt_mismatch_rows,
    sum(case when scoring_ts is null then 1 else 0 end) as scoring_ts_missing_rows,
    sum(case when snapshot_date is null then 1 else 0 end) as snapshot_date_missing_rows,
    sum(case when model_name is null or trim(model_name) = '' then 1 else 0 end) as model_name_missing_rows,
    sum(case when model_version is null or trim(model_version) = '' then 1 else 0 end) as model_version_missing_rows,
    sum(case when model_reference_path is null or trim(model_reference_path) = '' then 1 else 0 end) as model_reference_path_missing_rows,
    sum(case when selected_threshold is null then 1 else 0 end) as selected_threshold_missing_rows
from source_rows s
left join duplicate_player_ids d
    on s.player_id = d.player_id;

select
    player_id,
    duplicate_row_count
from duplicate_player_ids
order by duplicate_row_count desc, player_id
limit 100;
