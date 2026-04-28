set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420

select
    player_id,
    count(1) as duplicate_row_count
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}'
group by player_id
having count(1) > 1
order by duplicate_row_count desc, player_id
limit 100;
