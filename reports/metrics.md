# Model evaluation (20% test set)

Test set: 1,407 customers, churn rate 26.6%.

| model | recall | precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Rule: month-to-month contract | 87.2% | 41.4% | 0.561 |  |  |
| Logistic regression (no rebalancing) | 51.6% | 62.1% | 0.564 | 0.832 | 0.631 |
| Logistic regression + SMOTE (app) | 79.7% | 50.2% | 0.616 | 0.832 | 0.630 |
| Logistic regression, class_weight='balanced' | 79.1% | 49.7% | 0.610 | 0.832 | 0.628 |

Targeting with the app's model: the riskiest 20% of customers contain 50% of all churners, the riskiest 30% contain 65%.

## Highest-churn segments (full dataset)

| Contract | Internet | Customers | Churn rate |
|---|---|---|---|
| Month-to-month | Fiber optic | 2,128 | 54.6% |
| Month-to-month | DSL | 1,223 | 32.2% |
| One year | Fiber optic | 539 | 19.3% |
| Month-to-month | No | 524 | 18.9% |
| One year | DSL | 570 | 9.3% |

## Strongest model drivers (standardised coefficients)

Raise churn: TotalCharges (+0.89), InternetService_Fiber optic (+0.85), StreamingMovies_Yes (+0.35), StreamingTV_Yes (+0.28)

Lower churn: tenure (-1.59), MonthlyCharges (-1.21), Contract_Two year (-0.62), Contract_One year (-0.35)