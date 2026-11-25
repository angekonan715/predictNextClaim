# Data

The raw dataset is not stored in this repository. Download it yourself:

1. Go to **https://data.mendeley.com/datasets/sw4jmdb2sm** (DOI: 10.17632/sw4jmdb2sm.1)
2. Download the data file and the variable description file.
3. Put them in `data/raw/`.
4. Set `RAW_DATA_FILE` in `src/config.py` to the file name you downloaded.

The file used here is `Dataset of motor insurance portfolio.csv`. Fields are separated by semicolons. The companion file is `Descriptive of variables.csv`.

## Citation

Espinosa, Priscila; Lledó, Josep; Atance, David (2026), “A detailed dataset of motor insurance policies with coverage-specific financial information”, Mendeley Data, V1, doi: 10.17632/sw4jmdb2sm.1

Check the license shown on the Mendeley page before redistributing any part of the data.

## What a row is

Each row is one policy in one calendar year, from 2022, 2023, or 2024. There are 354,140 rows and 47 columns. After cleaning, 353,573 rows remain. The model trains on 2022–2023 (185,720 rows) and is tested on 2024 (167,853 rows).

There is no gender column. None of the 47 names is sex, gender, or a Spanish equivalent (`sexo`, `genero`).

## Variable notes

Codes below are the values stored in the file. The app shows a plain-language label for each code.

| Original column | Meaning | Used as | Notes |
|---|---|---|---|
| `insured_id` | An anonymous id for the policy. The same policy keeps the same id across years. | Identifier only | Not a predictor. |
| `year` | Calendar year of the row: 2022, 2023, or 2024. | Split only | Train on 2022–2023. Test on 2024. Not a predictor. |
| `total_exposure` | Fraction of the year the policy was in force. A full year is 1. A policy cancelled at mid-year is about 0.5. | Offset and feature | The GLM adds log(exposure). The other models read it as a column. The app sets it to 1, a full next year. Rows with exposure of 0 or below are dropped. |
| `total_claims` | Number of claims in that policy-year, all coverages added together. Most rows are 0. The largest value in the file is 18. | Target | `claim` is 1 when this is above 0, otherwise 0. About 14.4% of cleaned rows are 1. |
| `driver_age` | Age of the main driver, in years. | Numeric predictor | The file runs from 18 to 90. There is no driver under 18. The claim rate falls as age rises. |
| `vehicle_age` | Age of the car, in years. | Numeric predictor | Newer cars claim more often. Cars under 5 years claim about 31 times per 100 years on cover. Cars over 20 years claim about 19. |
| `age_driving_licence` | The data dictionary calls this the calendar year the licence was issued. The values run from 0 to 80, with a median of 20, so this project treats it as years the driver has held a licence. | Numeric predictor | Almost everyone sits between 10 and 40 years. Those two groups claim at the same rate, about 20 per 100 years on cover. |
| `vehicle_value` | Declared insured value of the car. | Numeric predictor | Capped at the 2022–2023 99th percentile, about 71,122. The same cap is applied to 2024. More expensive cars claim more often. |
| `seats` | Number of seats. | Left out | Most cars have 5 seats, and the claim rate barely moves. It was 0.6% of LightGBM's splits, so it is not a predictor. |
| `power_to_weight_ratio` | Approximate weight per horsepower (kg per hp). A higher number is a heavier, less powerful car. | Numeric predictor | The rate falls a little as this number rises. Rows at 0 or below are dropped. |
| `bonus_score` | The insurer's record of past claims. `G` good, `N` neutral, `B` poor. | Category | Good is about 20 claims per 100 years on cover, neutral about 28, poor about 34. About 96% of policies are already `G`. |
| `policy_type` | The coverage package. `TP` third party only, `TPG` third party plus glass, `CC` combined cover, `COMP_E` comprehensive with an excess, `COMP_N` comprehensive with no excess. | Category | Third party is about 10 per 100 years on cover. Comprehensive with no excess is about 41. Broader cover pays for more kinds of damage, so part of the gap is the cover itself. |
| `business_type` | `NB` a new policy, `P` a policy already on the books. | Category | The two rates are almost the same, about 20.2 and 20.4. |
| `payment_frequency` | How often the premium is billed. `A` once a year, `S` twice a year, `Q` every three months. | Category | Annual is about 20, twice a year about 22, quarterly about 29. This is a clue about the customer, not proof that the payment plan causes a claim. |
| `fuel_type` | `D` diesel, `G` gasoline. | Category | Missing fuel is filled as `Unknown` (1,287 rows). Diesel claims more often than gasoline (about 22 versus 17). |
| `vehicle_brand` | Brand name, already in capital letters (`RENAULT`, `BMW`, and so on). | Category | Brands with fewer than 1,000 policies in 2022–2023 are grouped into `Other`. 24 brands are kept. Among common brands the rate runs from about 13 (Rover) to about 27 (BMW). |
| `municipality_type` | Where the policyholder lives. `I` inland, `C` coast, `IS` islands. | Left out | Inland about 21, coast about 19, islands about 18 per 100 years on cover. It was 0.6% of LightGBM's splits, so it is not a predictor. |
| `circulation_area` | Where the car is usually driven. `U` urban (town), `R` rural (country). | Category | Town about 21, country about 19. |
| `policy_status` | `A` the policy was active, `C` it was cancelled, recorded at the time of analysis. | Left out | A cancellation can follow a claim, so this is not known for certain at the start of the year. |
| `total_premium` and each `*_premium` | The price the insurer already charged, in total and by coverage. | Left out | Using the price would mostly copy the insurer's current rating. |
| `*_claims` other than the target, and every `*_incurred` | Claim counts and claim costs for that same year, by coverage. | Left out | These are the outcome, or they are known only after a claim. |
| `liability_exposure` | Time on cover for liability only. | Left out | `total_exposure` already measures time on cover. |
