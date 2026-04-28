set odps.sql.type.system.odps2=true;
set odps.sql.hive.compatible=true;

create table if not exists pai_rec_prod.alg_uplift_phase1_response_scores_di (
    player_id string comment 'login_name-style player identifier',
    score_date date comment 'feature partition date scored by the Phase-1 observational response workflow',
    scoring_pt string comment 'source feature partition value used for scoring in yyyymmdd form',
    scoring_ts timestamp comment 'timestamp when the Python scoring workflow executed',
    snapshot_date date comment 'feature snapshot date from the source feature mart',
    predicted_response_score double comment 'predicted probability of positive 3-day observed response from the selected baseline model',
    score_rank bigint comment 'descending score rank within the scored partition',
    score_percentile double comment 'descending score percentile within the scored partition',
    response_priority_bucket string comment 'business-facing prioritization bucket derived from current-run score distribution',
    action_recommendation string comment 'simple transparent recommendation layer for targeting review',
    vip_level string comment 'decision-time vip segment from the source feature mart',
    model_name string comment 'saved baseline model family used for scoring',
    model_version string comment 'semantic model version if available; may be null when upstream artifacts do not persist one',
    model_reference_path string comment 'artifact reference path captured by the scoring workflow for reproducibility',
    selected_threshold double comment 'validation-selected threshold persisted with the scoring model artifact'
)
comment 'Phase-1 observational response scoring output for audience prioritization. Not a causal uplift or treatment-effect table.'
partitioned by (
    pt string comment 'scoring partition in yyyymmdd form; expected to align with scoring_pt'
)
lifecycle 90;
