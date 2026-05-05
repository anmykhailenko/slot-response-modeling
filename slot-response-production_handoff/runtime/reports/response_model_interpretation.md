
# Phase-1 Response Model Interpretation

This interpretation is for the selected baseline `lightgbm`. It explains who is historically more likely to respond positively after exposure in this observational dataset. It does not claim causal treatment effect.

## Feature Importance

| feature | importance_gain | importance_split |
| --- | --- | --- |
| recent_dep_cnt_7d | 2152648.02636528 | 557 |
| recency_last_bet_to_t | 575663.125492096 | 425 |
| recent_dep_amt_7d | 358481.81841373444 | 564 |
| recent_withdraw_cnt_7d | 313077.7307252884 | 308 |
| recent_win_amt_7d | 250638.05093240738 | 981 |
| recent_net_cash_in_7d | 213431.54539442062 | 635 |
| recent_withdraw_amt_7d | 180675.71332263947 | 199 |
| recent_bet_cnt_7d | 171530.87878227234 | 541 |
| recent_bet_days_7d | 134371.28087377548 | 335 |
| vip_level_V1 | 87325.80246973038 | 220 |
| recent_bet_amt_7d | 36351.00260257721 | 350 |
| vip_level_V4 | 33502.90838813782 | 182 |
| recent_ggr_amt_7d | 22797.9858584404 | 328 |
| recent_net_loss_amt_7d | 7988.836906909943 | 88 |
| vip_level_V2 | 5433.711681365967 | 83 |

## Score Deciles On Test Data

| decile | rows | positive_rate | avg_score |
| --- | --- | --- | --- |
| 10.0 | 142484.0 | 0.8636478481794447 | 0.8574511816393612 |
| 9.0 | 142483.0 | 0.7523985317546654 | 0.7445822882356414 |
| 8.0 | 142483.0 | 0.6686762631331457 | 0.6564694629997255 |
| 7.0 | 142483.0 | 0.5880280454510363 | 0.5827604456768128 |
| 6.0 | 142483.0 | 0.5112469557771805 | 0.5170298744834598 |
| 5.0 | 142484.0 | 0.44960837708093543 | 0.4638909475386571 |
| 4.0 | 142483.0 | 0.40104433511366266 | 0.4170414522399122 |
| 3.0 | 142483.0 | 0.35988854810749354 | 0.36859031054601393 |
| 2.0 | 142483.0 | 0.3121565379729512 | 0.3177717749381546 |
| 1.0 | 142484.0 | 0.22777294292692513 | 0.22597441682124825 |

## High-Score vs Low-Score Numeric Profile

| feature | high_score_mean | low_score_mean | difference |
| --- | --- | --- | --- |
| recent_bet_amt_7d | 21004.752327833477 | 477.45825411349773 | 20527.29407371998 |
| recent_ggr_amt_7d | 20595.562512746386 | 544.5094821761683 | 20051.05303057022 |
| recent_net_loss_amt_7d | 20595.562512746386 | 544.5094821761683 | 20051.05303057022 |
| recent_withdraw_amt_7d | 3672.474670662465 | 13.34245138016465 | 3659.1322192823004 |
| recent_bet_cnt_7d | 3806.758371174105 | 240.25378817122044 | 3566.5045830028844 |
| recent_dep_amt_7d | 3492.8594814820012 | 81.82893531158102 | 3411.03054617042 |
| recent_win_amt_7d | 409.18981508708754 | -67.05122806267063 | 476.2410431497582 |
| recent_dep_cnt_7d | 10.188278601657741 | 0.6535235782514405 | 9.5347550234063 |
| recent_withdraw_cnt_7d | 2.8881059494816927 | 0.01727223598604746 | 2.8708337134956454 |
| recent_bet_days_7d | 3.3102756118273757 | 1.1309980839819487 | 2.179277527845427 |
| pre_bet_days_30d | 0.0 | 0.0 | 0.0 |
| pre_win_amt_30d | 0.0 | 0.0 | 0.0 |

## High-Score vs Low-Score `vip_level` Mix

| vip_level | high_score_share | low_score_share |
| --- | --- | --- |
| V3 | 0.5202094284932237 | 0.28507962353403565 |
| V4 | 0.2986356267063439 | 0.04054518784697122 |
| V2 | 0.09106349529417544 | 0.09017567007993936 |
| V1 | 0.04841279310514237 | 0.5777004976032228 |
| V5 | 0.03570601405079904 | 0.0037057052420288736 |
| V6 | 0.003891692342244338 | 0.0013615659412000028 |
| V7 | 0.0010597755521711363 | 0.0005018142515247433 |
| V8 | 0.0010211744559000021 | 0.0004983050609546402 |
| UNKNOWN | 0.0 | 0.0004316304401226813 |

## Business Reading

- Users with the highest predicted positive-response scores are concentrated in higher pre-treatment betting and cash-in activity bands.
- Higher `vip_level` segments occupy a larger share of the high-score cohort, while `UNKNOWN`, missing, and lower VIP groups are more common among low-score users.
- The model therefore appears to be ranking historical engagement intensity and commercial value, which is reasonable for a first observational response baseline.
- Users with low recent betting, low recent deposits, and weaker 30-day betting history are much less likely to show a positive 3-day reaction.
- Weak or neutral responders are best interpreted as low-probability responders under historical exposure patterns, not as users who should never be treated.
