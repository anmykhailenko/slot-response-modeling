set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

-- Required runtime parameter:
--   ${scoring_pt}  e.g. 20260420

select
    count(1) as total_rows,
    sum(case when model_name is null or trim(model_name) = '' then 1 else 0 end) as model_name_missing_rows,
    sum(case when model_version is null or trim(model_version) = '' then 1 else 0 end) as model_version_missing_rows,
    sum(case when model_reference_path is null or trim(model_reference_path) = '' then 1 else 0 end) as model_reference_path_missing_rows,
    sum(case when selected_threshold is null then 1 else 0 end) as selected_threshold_missing_rows
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}';

select
    model_name,
    model_version,
    selected_threshold,
    count(1) as row_count
from pai_rec_prod.alg_uplift_phase1_response_scores_di
where pt = '${scoring_pt}'
group by model_name, model_version, selected_threshold
order by row_count desc, model_name, model_version;
