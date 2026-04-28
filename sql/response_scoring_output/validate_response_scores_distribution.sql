set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420

select
    response_priority_bucket,
    count(1) as row_count,
    count(1) / sum(count(1)) over () as population_share,
    min(predicted_response_score) as min_score,
    max(predicted_response_score) as max_score,
    avg(predicted_response_score) as avg_score
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}'
group by response_priority_bucket
order by avg_score;

select
    count(1) as total_rows,
    sum(case when predicted_response_score < 0.0 then 1 else 0 end) as score_below_zero_rows,
    sum(case when predicted_response_score > 1.0 then 1 else 0 end) as score_above_one_rows,
    sum(case when score_percentile < 0.0 then 1 else 0 end) as percentile_below_zero_rows,
    sum(case when score_percentile > 1.0 then 1 else 0 end) as percentile_above_one_rows,
    sum(case when response_priority_bucket not in ('very_low', 'low', 'medium', 'high', 'very_high') then 1 else 0 end) as invalid_bucket_rows
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}';
