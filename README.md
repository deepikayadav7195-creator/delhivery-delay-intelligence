# Delhivery Last-Mile Delay Intelligence

## Project Overview

This project focuses on identifying shipments that may get delayed during last-mile delivery. It uses shipment details, hub information and machine learning models to predict delays and suggest actions that can help the operations team.

## 1. How to Run

First, install the required libraries:

```bash
python3 -m pip install -r requirements.txt
```

Run the Python files in this order:

```bash
python3 generate_data.py
python3 part_a_clean_eda.py
python3 part_b_regression.py
python3 part_c_classification.py
python3 part_d_segments.py
python3 package_model.py
python3 predict_delay.py
```

The generated datasets are stored in the `data` folder. Model files are saved in `models`, and analysis results and predictions are saved in `outputs`.

## 2. Data Findings

The original shipment dataset contained 1,224 rows. After removing 24 duplicate records, 1,200 rows remained for analysis.

During data cleaning, missing values were found in distance, parcel weight, declared value and payment mode. These were handled using suitable filling methods. Five records also had unrealistic parcel weight values, which were treated as missing values before filling them.

Out of 1,200 shipments, 278 were delayed, giving an overall delay rate of 23.17%.

**Delay patterns**

- Express shipments had the highest delay rate at 48.97%, followed by Standard at 20.56% and Economy at 8.57%.
- Delay rates increased with hub tier. Tier 1 had a delay rate of 4%, Tier 2 had 21.50%, and Tier 3 had 44%.
- Longer distances were associated with more delays. The delay rate increased from 2.19% for shipments up to 200 km to 85.23% for shipments above 600 km.

**Correlation findings**

The three strongest correlations with delivery time were:

| Feature | Correlation |
|---|---:|
| promised_hours | 0.8372 |
| distance_km | 0.4013 |
| n_stops | 0.3247 |

Promised delivery hours had the strongest positive correlation with actual delivery time. Distance and the number of stops also showed positive relationships with delivery time.

**Declared value**

The mean declared value was ₹1,533.05, while the median was ₹1,121.50. Since the mean was higher than the median, the distribution was right-skewed. I applied a log transformation to reduce the effect of very high declared values. After transformation, the mean and median were much closer, at 7.0277 and 7.0224 respectively.

## 3. Model Results – Regression

I tested four regression models to predict delivery time: Linear Regression, Decision Tree, Random Forest and Gradient Boosting. I also used a baseline model for comparison.

**Model comparison**

The baseline model had a validation RMSE of 18.8552 hours and an R² of -0.0212. This means it performed worse than simply predicting the average delivery time.

Linear Regression gave the best validation result, with an RMSE of 4.8912 hours and an R² of 0.9313. Gradient Boosting was very close, with an RMSE of 4.9165 hours. Random Forest and Decision Tree had validation RMSE values of 5.4784 and 6.2796 hours respectively.

**Decision Tree depth analysis**

I tested different maximum depths for the Decision Tree. Depth 6 gave the lowest validation RMSE of 6.2796 hours.

At smaller depths, the model had higher training and validation errors, which suggests underfitting or high bias. At greater depths, training error continued to decrease while validation error increased. The unlimited-depth tree had zero training RMSE but a validation RMSE of 7.7890 hours. This indicates overfitting and higher variance.

**Gradient Boosting learning rate**

I tested learning rates of 0.01, 0.10 and 1.00. The best result came from 0.10, with a validation RMSE of 4.9165 hours. A rate of 0.01 learned too slowly in this setup, while 1.00 gave a worse result.

**Cross-validation**

Cross-validation gave the following mean RMSE results:

| Model | Mean CV RMSE |
|---|---:|
| Linear Regression | 5.1578 |
| Gradient Boosting | 4.8326 |

Cross-validation slightly favoured Gradient Boosting, while the validation comparison favoured Linear Regression. I selected Linear Regression because it had the lowest validation RMSE and performed well on the test set.

**Final test results**

- Test RMSE: 4.4932 hours
- Test MAE: 3.5376 hours
- Test R²: 0.9499

The final results show that Linear Regression predicted delivery time with a relatively low error on the test data.

## 4. Model Results – Classification

The classification models were used to predict whether a shipment was likely to be delayed. I tested Logistic Regression, Decision Tree and Random Forest, along with a baseline that predicted every shipment as on time.

**Initial model comparison**

The baseline achieved an accuracy of 76.67%, but its recall and F1-score were both zero because it failed to identify any delayed shipments.

In the initial model comparison on the test split, Logistic Regression achieved an F1-score of 0.8046. The default Random Forest achieved 0.8000, while the Decision Tree achieved 0.7470. These results helped compare the initial classification models before the final feature selection and threshold evaluation.

**Feature importance and noise columns**

The feature analysis showed that distance, hub tier, hub capacity and number of stops were among the most useful features. I also checked five bookkeeping columns: label_print_batch, scanner_build_no, tape_roll_no, sticker_serial_mod and shelf_slot_no. Together, these five columns contributed approximately 22.52% of the Random Forest importance. However, none ranked among the top seven features based on F-scores, so I excluded them from the final feature selection.

**Final model and threshold**

The final model uses seven selected features and a probability threshold of 0.5. This threshold was selected because it achieved a validation recall of 0.8333 while meeting the target recall of at least 0.80.

**Final test results**

- Accuracy: 86.67%
- Precision: 68.89%
- Recall: 75.61%
- F1-score: 72.09%

The model correctly identified 31 delayed shipments out of 41 in the test set. It also flagged 14 shipments that were not delayed.

These results show that the model can help identify shipments that may need attention before delivery problems occur.

## 5. Operations Recommendation

I used K-Means clustering to divide the 24 hubs into three operational groups: Standard, Watch and Escalate. I compared different cluster numbers (K=2, 3, 4 and 5). Although K=4 and K=5 gave higher separation ratios, I selected K=3 because it provides three practical groups for the operations team and avoids creating very small clusters.

**Hub segment results**

| Segment | Hubs | Average delay rate |
|---|---:|---:|
| Standard | 7 | 2.57% |
| Watch | 8 | 15.50% |
| Escalate | 9 | 46.00% |

The Standard group has a low delay rate and can follow normal operations. The Watch group needs regular monitoring, while the Escalate group needs more attention from hub managers because its delay rate is much higher.

**Shipment-level actions**

The final classification model uses a probability threshold of 0.5 to flag shipments as potentially delayed. The action also depends on the hub segment.

- No action: 40 shipments
- Escalate to hub manager: 6 shipments
- Proactive customer call: 2 shipments

Out of 48 new shipments, eight were flagged for attention. Six were assigned to hub manager escalation and two to proactive customer calls.

This approach helps the operations team focus on shipments that may need intervention instead of treating every shipment in the same way.

The comparison between hub tier and K-Means segmentation showed four mismatches. HUB07 (Kolkata) was placed in the Watch segment instead of Standard. HUB13 (Nagpur) and HUB15 (Kochi) were placed in Escalate instead of Watch. HUB22 (Dehradun) was placed in Watch instead of Escalate. This shows that capacity-based hub tiers do not always match actual delivery performance. Therefore, operational decisions should also consider delivery hours and delay rates.

## 6. Production Plan

The trained model is saved in the `models/` folder and can be used to predict delays for new shipments. The `predict_delay.py` script loads the saved model, processes new shipment data and generates delay predictions.

The model uses a probability threshold of 0.5 to flag shipments that may be delayed. Based on the prediction and hub segment, the system recommends actions such as escalating the case to a hub manager or making a proactive customer call.

In a real-world setup, this process can be connected to the shipment tracking system and run regularly. The model should also be monitored using new delivery data. If its performance decreases or delivery patterns change, it can be retrained with updated data.

## 7. Limitations

- The project uses generated shipment data, so actual delivery conditions may be different from the patterns found in this dataset.
- The model depends on the quality and accuracy of shipment and hub information.
- Unexpected situations such as heavy rain, traffic and operational disruptions can affect delivery times.
- The model may sometimes predict delays incorrectly, so important cases should be reviewed by the operations team.
- The model has not yet been tested on real company shipment data. Its performance should be validated using real data before production deployment.
- The model may need regular updates as delivery patterns, routes and operational conditions change.
