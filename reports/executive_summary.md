# Executive summary

**Predicting next-year car insurance claims**

The question is which drivers are most likely to file at least one car insurance claim next year. The data are 354,140 anonymised policy-years from a Spanish insurer, 2022 through 2024. The model learns from 2022 and 2023 and is tested on 2024, the way an insurer would use it on a year it has not seen.

A row is counted as a claim if `total_claims` is above zero. About 14% of training policies and 15% of 2024 policies had a claim. Only facts known on the first day of the year go into the model: the driver, the car, the coverage, where the car is driven, and how much of the year the policy was in force. Premiums, claim costs, and whether the policy was later cancelled stay out, because those are known later or are the insurer's existing price.

Six models were compared. LightGBM is the best at sorting who will claim (AUC 0.692 on 2024) and its percentages sit close to what actually happened (calibration gap 0.003). XGBoost is next at 0.683. A GLM and a random forest both reach 0.679, but the forest's percentages are farther from the real rate. The app uses LightGBM.

About half of LightGBM's splits use time on cover, because a short policy has less time to claim. The app asks about a full next year, so that input is fixed at one year. After that, coverage matters most, then vehicle value and vehicle age. In the raw rates, drivers aged 18–25 claim about 32 times per 100 years on cover, and drivers over 65 claim about 17. A poor past-claims record claims about 34 times, against about 20 for a good record.

The file has no gender column, so the model does not use gender and a gap between women and men cannot be measured. The predictions are a study tool. They are not an insurance quote.

The app is at https://angekonan715-predictnextclaim-appapp-vz9rsh.streamlit.app/
