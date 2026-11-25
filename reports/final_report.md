# Predicting next-year car insurance claims

**Ange Konan** · Senior project · Data Science major, Statistics minor

## 1. The question

Which drivers are most likely to file at least one car insurance claim next year?

An insurer already knows a few facts when a policy year starts: who drives, what car, what cover, and where the car is driven. This project turns those facts into a probability. The probability is the chance of at least one claim over a full next year. It is a study result, not a price and not an insurance quote.

## 2. The data

The data are anonymised motor policies from a Spanish insurer. Espinosa, Priscila; Lledó, Josep; Atance, David (2026), “A detailed dataset of motor insurance policies with coverage-specific financial information”, Mendeley Data, V1, doi: 10.17632/sw4jmdb2sm.1. Each row is one policy in one calendar year. The file has 354,140 rows, 47 columns, and the years 2022, 2023, and 2024.

There is no 2025 or 2026 outcome in the file. "Next year" is handled by training on 2022–2023 and testing on 2024. That is an out-of-time test: the model is judged on a year it did not learn from.

The raw file is not in the git repository. Column-by-column notes are in `data/README.md`.

## 3. The target

The question is yes or no: did this policy have at least one claim?

`total_claims` adds every coverage together. About 86% of policy-years have zero claims. The rest have one or more, up to 18. A count model would spend most of its effort on the zeros. The target used here is

`claim = 1` when `total_claims > 0`, and `0` otherwise.

About 13.6% of the 2022–2023 rows have a claim. About 15.1% of the 2024 rows do.

The rate in the charts is not a simple average of that 0/1 flag. It is claims divided by years on cover (`total_exposure`). A policy that lasted half a year counts as half a policy. That stops a short policy from looking as risky as a full year.

## 4. What the model is allowed to see

A column is used only if the insurer would already know it on the first day of that policy year.

Kept: driver age, years with a licence, vehicle age, vehicle value, weight per horsepower, fuel, brand, coverage, new versus existing policy, payment schedule, past-claims record, and town versus country. Time on cover is included so a partial year is not treated as a full year. Seats and home area were tried and then left out. Each was 0.6% of the splits, and the claim rate barely moved.

Left out:

- Claim counts and claim costs for that same year. Those are the outcome.
- `policy_status` (active or cancelled). A cancellation can follow a claim.
- Premiums. Those are the insurer's current price. Using them would mostly copy the existing rating.
- The policy id and the calendar year. The year is used only to split train from test.

Cleaning drops rows the model cannot use: no time on cover, a missing age or value, a driver under 16, a negative age, or a weight-per-horsepower of 0 or below. Missing fuel is kept as its own group, `Unknown`. That leaves 353,573 of 354,140 rows.

Two adjustments are learned on 2022–2023 only, then applied to 2024, so the test year does not help choose them:

- Vehicle value is capped at the training 99th percentile, about 71,122. One very expensive car would otherwise pull the rate.
- Brands with fewer than 1,000 training policies are grouped into `Other`. Twenty-four brands are kept.

No new calculated columns were added. The ages are already in the file. The data dictionary calls `age_driving_licence` the calendar year the licence was issued, but the values run from 0 to 80 with a median of 20, so the project treats them as years the person has held a licence.

The file's youngest driver is 18. The app slider starts at 16 so a 16- or 17-year-old can be typed in, but no such driver is in the training file. A score for age 16 is an extrapolation.

## 5. What the claim rates show

The chart is `reports/figures/claim_rates.png`. Rates are claims per year on cover.

- **Driver age.** Drivers 18–25 claim about 32 times per 100 years on cover. Drivers over 65 claim about 17. The rate falls as age rises.
- **Vehicle age.** Cars under 5 years claim about 31 times. Cars over 20 years claim about 19. Newer cars claim more often.
- **Vehicle value.** From 8,000 to 15,000 the rate is about 15. Above 40,000 it is about 26. More expensive cars claim more often.
- **Past claims.** A good record is about 20, a neutral record about 28, and a poor record about 34. About 96% of policies are already in the good group, so this score separates only a few policies.
- **Coverage.** Third party only is about 10. Combined cover is about 17. Comprehensive cover with no excess is about 41. Broader cover pays for more kinds of damage, so part of that gap is the cover, not only the driver.
- **Payment.** Once a year is about 20, twice a year about 22, and every three months about 29. Paying more often goes with a higher claim rate. That is a clue about the customer. It is not proof that the payment plan causes a claim.

A few facts barely move. A new policy and an existing policy are about 20.2 and 20.4. Town is about 21 and country about 19. Years with a licence sit mostly between 10 and 40, and both groups claim about 20 times per 100 years on cover.

## 6. The models

Six models see the same training rows and the same test year.

| Model | Idea |
|---|---|
| GLM | A formula. The link is cloglog and log(exposure) is an offset. If claims arrive at rate λ, the chance of at least one claim over time *t* is 1 − exp(−λt). The offset adds log(*t*) instead of estimating it. |
| Logistic regression | One straight line for the log-odds. Numeric inputs are scaled so vehicle value does not drown the ages. |
| Decision tree | At most four yes-or-no questions. The score is the claim rate of the policies that follow the same answers. |
| Random forest | 200 trees, each at most 8 questions deep. The score is their average. |
| XGBoost | 300 shallow trees. Each new tree corrects the policies the earlier trees got wrong. Learning rate 0.05. |
| LightGBM | The same idea as XGBoost, with LightGBM's way of choosing the splits. 300 trees, learning rate 0.05, up to 31 leaves. |

The tree settings were fixed before looking at 2024. There is no search over 2024, and there is no oversampling. The data are already a yes/no with a known base rate of about 15%. Oversampling would change that base rate and make the percentages less honest.

Exposure is an offset in the GLM. In the other five models it is an ordinary column.

## 7. Results on 2024

AUC is a sorting score. A model with AUC 0.50 is a coin flip. A model with AUC 1 puts every driver who claimed above every driver who did not. Gini is 2 × AUC − 1, the same idea on a scale insurers often use. The calibration gap is how far the predicted percentages sit from the claim rates that actually happened. Smaller is better.

| Model | AUC | Gini | Calibration gap |
|---|---|---|---|
| LightGBM | 0.692 | 0.384 | 0.003 |
| XGBoost | 0.683 | 0.366 | 0.003 |
| Random forest | 0.679 | 0.359 | 0.019 |
| GLM | 0.679 | 0.359 | 0.005 |
| Logistic regression | 0.676 | 0.352 | 0.008 |
| Decision tree | 0.663 | 0.326 | 0.005 |

The chart is `reports/figures/model_comparison.png`.

LightGBM wins. Its lead over XGBoost is 0.009 AUC, which is enough to keep it. The random forest matches the GLM on sorting but its probabilities are the worst of the six (gap 0.019), so it loses the tie-break. A simpler model would have been chosen only if it was within 0.005 AUC and its calibration gap was within 0.002 of the best close model. None was.

The lift chart (`reports/figures/lift.png`) splits 2024 into ten groups from lowest predicted risk to highest. The bottom group claims about 3 times per 100 policies. The top group claims about 35. The predicted line sits on the actual line.

The calibration chart (`reports/figures/calibration.png`) plots the predicted percentage against the observed claim rate for the GLM and for LightGBM. Both stay close to the diagonal. A predicted 20% really happens about 20% of the time.

## 8. What LightGBM uses

LightGBM adds hundreds of small trees. Each new tree corrects the policies the earlier trees scored too high or too low. That is why it can follow a bend, such as "young drivers claim more, and the gap shrinks after 35," which a single straight line handles poorly.

Gain measures how much each fact contributed to the splits. One-hot columns are added back together, so "coverage" is one bar rather than one bar per coverage code. The chart is `reports/figures/feature_importance.png`.

| Fact | Share of splits |
|---|---|
| Time on cover | 53% |
| Coverage | 15% |
| Vehicle value | 6% |
| Vehicle age | 5% |
| Weight per horsepower | 5% |
| Driver age | 4% |
| Payment schedule | 3% |
| Years with a licence | 2% |
| Brand | 2% |
| Fuel, past claims, town or country, new or existing policy | about 1% or less each |

Time on cover dominates because a policy that was active for a short time has less time to claim. The app fixes that input at one full year. The facts that then change from one driver to the next start with coverage, vehicle value, and vehicle age.

SHAP answers a different question: for one policy, how much did each fact push the score up or down? The plot `reports/figures/shap_summary.png` uses 2,000 policies from 2024. A point to the right raised that policy's chance. High time on cover pushes the chance up. Low time on cover pushes it down, sometimes by a lot. Comprehensive cover with no excess pushes the chance up when the policy has that cover. Driver age moves individual scores a little in both directions, which matches its smaller share of the splits.

The GLM is still the model to read when the goal is one formula rather than a score. LightGBM is the model in the app because it sorts better and its percentages stay honest.

## 9. Fairness

Since 2012, insurers in the European Union cannot use gender to set a price. Gender is not one of the features in this model. The check was whether the file even contains it, so a with-versus-without comparison could be run.

It does not. None of the 47 columns is sex, gender, `sexo`, or `genero`. The model cannot be refit with and without gender, and a difference in scores for women and men cannot be measured from this file.

Other columns can still carry a social pattern. Past claims and payment schedule move the claim rate a little. They stay in the model because they are known on the first day of the year and they describe the policy. They are not a stand-in that was built to recover gender. Without a gender column, that recovery cannot be tested here. Home area was left out: inland, coast, and islands differ by only about 3 claims per 100 years on cover.

## 10. The app

The Streamlit app asks for the day-one facts and returns a full-year probability and a Low, Medium, or High label. Low is the lower third of full-year scores on the 2024 policies, Medium the middle third, and High the upper third. The cutoffs are about 14.1% and 19.6%.

A sample driver (age 47, 20 years licensed, a 24-year-old Renault worth 25,000, combined cover) scores about 15.3%, Medium. The same inputs at age 16 score about 17.1%, still Medium. Age 16 is allowed on the form. It is not in the training file.

The last thing on the page is the gain chart: time on cover, then coverage, then vehicle value.

Live app: https://angekonan715-predictnextclaim-appapp-vz9rsh.streamlit.app/

## 11. Limits

- AUC 0.692 is a real improvement on a coin flip. It is not a sharp separator. Many drivers who claim still sit among drivers who do not.
- The test is one insurer and one country, for three years. A different market can have a different rate.
- Associations are not causes. Quarterly payment goes with more claims. The chart does not say the payment plan creates the claims.
- Broader coverage goes with more claims partly because that coverage pays for more kinds of damage.
- Age 16 and 17 can be entered in the app, and the file has no such driver.
- There is no gender field, so a fairness gap by gender cannot be measured.
- The model was not tuned by searching settings on 2024. A search inside 2022–2023 might move the AUC by a small amount. It would not change the decision to keep a model whose percentages match the observed rate.
