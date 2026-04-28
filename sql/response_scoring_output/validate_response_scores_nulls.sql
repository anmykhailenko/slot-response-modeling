set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420

select
    count(1) as total_rows,
    sum(case when player_id is null or trim(player_id) = '' then 1 else 0 end) as player_id_null_or_blank_rows,
    sum(case when score_date is null then 1 else 0 end) as score_date_null_rows,
    sum(case when scoring_pt is null or trim(scoring_pt) = '' then 1 else 0 end) as scoring_pt_null_or_blank_rows,
    sum(case when scoring_ts is null then 1 else 0 end) as scoring_ts_null_rows,
    sum(case when snapshot_date is null then 1 else 0 end) as snapshot_date_null_rows,
    sum(case when predicted_response_score is null then 1 else 0 end) as predicted_response_score_null_rows,
    sum(case when score_rank is null then 1 else 0 end) as score_rank_null_rows,
    sum(case when score_percentile is null then 1 else 0 end) as score_percentile_null_rows,
    sum(case when response_priority_bucket is null or trim(response_priority_bucket) = '' then 1 else 0 end) as response_priority_bucket_null_or_blank_rows,
    sum(case when action_recommendation is null or trim(action_recommendation) = '' then 1 else 0 end) as action_recommendation_null_or_blank_rows,
    sum(case when model_name is null or trim(model_name) = '' then 1 else 0 end) as model_name_null_or_blank_rows,
    sum(case when selected_threshold is null then 1 else 0 end) as selected_threshold_null_rows
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}';
