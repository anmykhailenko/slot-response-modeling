
# Response Model Threshold Recommendation

The threshold grid does not justify hiding weak VIP segments. It does support a clearer operating-point discussion.

## Candidate Global Operating Views On Test

| view | threshold | global_precision | global_recall | global_f1 | global_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- |
| recall-prioritized | 0.3 | 0.5510629980933843 | 0.9442272573184467 | 0.6959567902167441 | 0.8797732786930117 |
| balanced | 0.5 | 0.6852652144325824 | 0.640059269303531 | 0.661891269362799 | 0.47957550112890424 |
| precision-protected | 0.6 | 0.7502245910503967 | 0.4760093278073638 | 0.5824564103121631 | 0.3257764243248156 |

## Interpretation

- `recall-prioritized` view at `0.30`: global recall rises to `0.9442`, but predicted-positive rate expands to `0.8798`. For `V1`, this meaningfully improves recall to `0.8436` and F1 to `0.5872`, but precision stays modest at `0.4503`.
- `balanced` view at `0.50`: this is the recommended default because it was chosen on validation under the documented precision-with-recall-floor policy rather than on test. It gives global precision `0.6853` and recall `0.6401`. For `V2`, it stays close to the segment’s best observed F1 on the preferred grid.
- `precision-protected` view at `0.60`: global precision improves to `0.7502`, while recall falls to `0.4760`. This hurts `V1` sharply, dropping its recall to `0.1613` and F1 to `0.2616`.

## Recommendation

- Keep one documented global threshold as the default near the current validation-selected point `0.50`. It is not segment-optimal, but it is transparent and avoids introducing silent policy complexity.
- Use the threshold grid to show that `V1` is the main exception. `V1` prefers a materially lower threshold than the global operating point, but even that lower threshold does not make the segment strong.
- Treat `V2` as acceptable for a global-threshold ranking view, but do not describe it as strong segment quality.
- VIP-aware thresholding can be considered later as an explicit policy-layer experiment. It should only be introduced after business review, because it changes who gets contacted by segment and should not be framed as a hidden metric fix.
