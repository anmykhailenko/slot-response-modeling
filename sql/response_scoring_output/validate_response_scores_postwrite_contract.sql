set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420
--
-- Purpose:
--   Mandatory validation after overwriting the final table partition.
--   A healthy publish returns zero for every violation metric below.

with final_partition as (
    select
        player_id,
        score_date,
        scoring_pt,
        scoring_ts,
        snapshot_date,
        predicted_response_score,
        score_rank,
        score_percentile,
        response_priority_bucket,
        action_recommendation,
        vip_level,
        model_name,
        model_version,
        model_reference_path,
        selected_threshold,
        pt
    from pai_rec_prod.alg_uplift_phase1_response_scores_di
    where pt = '${scoring_pt}'
),
duplicate_player_ids as (
    select
        player_id,
        count(1) as duplicate_row_count
    from final_partition
    group by player_id
    having count(1) > 1
)
select
    '${scoring_pt}' as scoring_pt,
    count(1) as row_count,
    count(1) - count(distinct player_id) as duplicate_player_id_rows,
    count(distinct d.player_id) as duplicate_player_id_groups,
    sum(case when player_id is null or trim(player_id) = '' then 1 else 0 end) as player_id_missing_rows,
    sum(case when score_date is null then 1 else 0 end) as score_date_missing_rows,
    sum(case when scoring_pt is null or trim(scoring_pt) = '' then 1 else 0 end) as scoring_pt_missing_rows,
    sum(case when scoring_pt <> '${scoring_pt}' then 1 else 0 end) as scoring_pt_mismatch_rows,
    sum(case when scoring_ts is null then 1 else 0 end) as scoring_ts_missing_rows,
    sum(case when snapshot_date is null then 1 else 0 end) as snapshot_date_missing_rows,
    sum(case when predicted_response_score is null then 1 else 0 end) as score_missing_rows,
    sum(case when predicted_response_score < 0.0 or predicted_response_score > 1.0 then 1 else 0 end) as score_out_of_range_rows,
    sum(case when response_priority_bucket is null or trim(response_priority_bucket) = '' then 1 else 0 end) as bucket_missing_rows,
    sum(case when response_priority_bucket not in ('very_low', 'low', 'medium', 'high', 'very_high') then 1 else 0 end) as invalid_bucket_rows,
    sum(case when action_recommendation is null or trim(action_recommendation) = '' then 1 else 0 end) as action_recommendation_missing_rows,
    sum(case when model_name is null or trim(model_name) = '' then 1 else 0 end) as model_name_missing_rows,
    sum(case when model_version is null or trim(model_version) = '' then 1 else 0 end) as model_version_missing_rows,
    sum(case when model_reference_path is null or trim(model_reference_path) = '' then 1 else 0 end) as model_reference_path_missing_rows,
    sum(case when selected_threshold is null then 1 else 0 end) as selected_threshold_missing_rows,
    sum(case when pt <> '${scoring_pt}' then 1 else 0 end) as partition_mismatch_rows
from final_partition f
left join duplicate_player_ids d
    on f.player_id = d.player_id;

select
    player_id,
    duplicate_row_count
from duplicate_player_ids
order by duplicate_row_count desc, player_id
limit 100;
