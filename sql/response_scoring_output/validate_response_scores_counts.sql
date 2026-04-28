set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420

select
    pt,
    count(1) as row_count,
    count(distinct player_id) as distinct_player_id_count,
    count(1) - count(distinct player_id) as duplicate_player_id_rows,
    min(score_date) as min_score_date,
    max(score_date) as max_score_date,
    min(snapshot_date) as min_snapshot_date,
    max(snapshot_date) as max_snapshot_date
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}'
group by pt;
