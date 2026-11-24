# 🏨 INN Hotels Group: Predicting Booking Cancellations

A classification project that finds what drives booking cancellations at INN Hotels Group and predicts which bookings are likely to be cancelled. It uses 36,275 bookings and 17 predictors, and compares a logistic regression with four decision trees.

The dataset and business case come from the Great Learning Postgraduate Program in Data Science.

## Results at a glance

- **Final model:** a post-pruned decision tree, tested on a held-out 30% (10,883 bookings).
- **Test performance:** accuracy 88.3% (train 90.2%).
- **On cancellations:** the model finds 2,789 of 3,522 actual cancellations (79.2% recall). When it predicts a cancellation, it is right 83.9% of the time.
- **Baseline:** logistic regression reached 80.3% test accuracy and ROC-AUC 0.85.
- **Top predictor:** lead time (about 40% of feature importance), then average room price, market segment and number of special requests.
- **Cancellation rate:** 32.8% of all bookings. Repeat guests cancel only 1.7% of the time.
- **Caveat:** the pruning strength was chosen using the test set, so 88.3% is slightly optimistic. See Limitations.

## Business question

A high number of cancellations costs INN Hotels revenue and makes room planning harder. The hotel wants to know what drives cancellations and to flag risky bookings early, so it can set cancellation and refund policies by risk level.

## Data

36,275 rows, 19 columns. No missing values, no duplicate rows.

| Group | Variables |
|---|---|
| Booking ID and target | `Booking_ID`, `booking_status` (Canceled / Not_Canceled) |
| Guests | `no_of_adults`, `no_of_children`, `repeated_guest`, `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled` |
| Stay | `no_of_weekend_nights`, `no_of_week_nights`, `type_of_meal_plan`, `room_type_reserved`, `required_car_parking_space`, `avg_price_per_room`, `no_of_special_requests` |
| Timing | `lead_time`, `arrival_year`, `arrival_month`, `arrival_date` |
| Channel | `market_segment_type` |

## Method

1. Checked shape, data types, missing values, duplicates and distinct values.
2. Univariate and bivariate analysis of every variable, plus six business questions.
3. Encoded categorical variables as numbers and set the target to Canceled = 0, Not_Canceled = 1.
4. Split the data 70/30 (25,392 train, 10,883 test).
5. Logistic regression: checked VIF (1.01 to 1.74), removed five variables by backward elimination at the 5% level, then evaluated on the test set.
6. Four decision trees: unrestricted, depth limited to 3, tuned with GridSearchCV, and post-pruned with cost-complexity pruning.
7. Compared all models and picked the post-pruned tree.

## Findings

### Exploratory analysis

- 32.8% of bookings are cancelled (11,885 of 36,275).
- October is the busiest month, then September and August. January is the quietest.
- Online bookings are 64% of the total, Offline 29%. Corporate, Complementary and Aviation are small.
- Online has the highest median room price (about 107), Corporate is lower (about 79), and Complementary is 0.
- Cancelled bookings had a median lead time of 122 days, against 39 days for kept bookings.

| Lead time | Bookings | Cancelled |
|---|---|---|
| 0 to 30 days | 12,411 | 14.2% |
| 31 to 90 days | 10,815 | 25.3% |
| 91 to 180 days | 7,772 | 44.9% |
| 181+ days | 5,277 | 73.9% |

| Special requests | Bookings | Cancelled |
|---|---|---|
| 0 | 19,777 | 43.2% |
| 1 | 11,373 | 23.8% |
| 2 | 4,364 | 14.6% |
| 3 or more | 761 | 0 |

| Market segment | Bookings | Cancelled |
|---|---|---|
| Online | 23,214 | 36.5% |
| Offline | 10,528 | 29.9% |
| Corporate | 2,017 | 10.9% |
| Aviation | 125 | 29.6% |
| Complementary | 391 | 0% |

Repeat guests: 16 of 930 bookings were cancelled (1.7%), against 33.6% for first-time guests.

### Model comparison

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Logistic regression | not recorded | 80.26% |
| Decision tree, unrestricted | 99.42% | 86.97% |
| Decision tree, depth 3 | 78.44% | 79.13% |
| Decision tree, grid-search tuned | 82.42% | 82.75% |
| **Decision tree, post-pruned (final)** | **90.16%** | **88.33%** |

The unrestricted tree memorises the training data (99.4% train vs 87.0% test). The depth-3 tree is too simple. Pruning gave the best balance.

### Final model, test confusion matrix

| | Predicted cancelled | Predicted not cancelled |
|---|---|---|
| **Actual cancelled** | 2,789 | 733 |
| **Actual not cancelled** | 537 | 6,824 |

## Recommendations

- Use lead time as the main risk signal. Long-lead bookings need tighter terms.
- Replace one cancellation policy with risk tiers: free cancellation for low risk, a limited free window for medium risk, and a deposit or stricter terms for high risk.
- Keep investing in repeat guests through loyalty offers, since they almost never cancel.

## Limitations

- The pruning strength (`ccp_alpha`) was picked by test accuracy, not by a separate validation set or cross-validation. The 88.3% is therefore a little optimistic.
- Market segment, room type and meal plan are coded as numbers (for example Online = 1, Offline = 2) and treated as ordered in the logistic regression. Its coefficients for these variables should not be read as effects.
- The target is coded Canceled = 0, so the "recall" printed in the notebook is for Not_Canceled. The cancellation recall above comes from the confusion matrix.
- The data covers 2017 and 2018 only. The cancellation rate was 14.8% for 2017 arrivals and 36.7% for 2018 arrivals, so patterns may not hold if demand or policy changes.
- All findings are associations, not causes.

## Repository contents

| File | Description |
|---|---|
| `Inn_Hotels_Group.ipynb` | Full analysis: EDA, logistic regression, four decision trees |
| `INNHotelsGroup.csv` | Data File |
| `README.md` | This file |

## How to run

The dataset (`INNHotelsGroup.csv`) is not included in this repository. The notebook reads it from a Google Drive path, so change the loading line to `pd.read_csv("INNHotelsGroup.csv")` and place the file next to the notebook.

```
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
jupyter notebook Inn_Hotels_Group.ipynb
```

## Tools

Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn

## Author

Dhruv Sai Gedda | [LinkedIn](https://www.linkedin.com/in/dhruv-sai) | [Portfolio](https://dsai04.github.io/)
