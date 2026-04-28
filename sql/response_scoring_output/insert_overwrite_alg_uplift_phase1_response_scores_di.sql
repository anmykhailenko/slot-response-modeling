set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameters:
--   ${direct_scored_source}  relation name bound for this run; prefer a temporary or transient ODPS view
--                            that exposes the scored output rows from the Python scoring workflow
--   ${scoring_pt}            e.g. 20260420
--
-- Direct-write contract:
--   1. run the Python scoring workflow outside SQL
--   2. expose the scored rows to ODPS as `${direct_scored_source}` for the current session
--   3. run `validate_response_scores_prewrite_contract.sql`
--   4. only if every pre-write check passes, run this `INSERT OVERWRITE`
--   5. run the post-write validation SQL against the final partition
--
-- Production-safety rules:
--   - this statement overwrites exactly one final-table partition: `pt = '${scoring_pt}'`
--   - no append behavior is allowed in this package
--   - one row per `player_id` in the partition must be guaranteed by pre-write validation
--   - do not run this statement if pre-write validation returns any violation
--
-- Required source columns:
--   player_id
--   score_date
--   scoring_pt
--   scoring_ts
--   snapshot_date
--   predicted_response_score
--   score_rank
--   score_percentile
--   response_priority_bucket
--   action_recommendation
--   vip_level
--   model_name
--   model_version
--   model_reference_path
--   selected_threshold

insert overwrite table pai_rec_prod.alg_uplift_phase1_response_scores_di partition (pt = '${scoring_pt}')
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
where cast(scoring_pt as string) = '${scoring_pt}';
