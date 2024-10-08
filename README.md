# Diabetes Prediction Using Machine Learning

This project aims to predict the likelihood of an individual having diabetes based on various health indicators using two machine learning models: **Random Forest** and **Logistic Regression**. The dataset used comes from the CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS), with health-related features such as BMI, smoking status, and physical activity.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Visualizations](#visualizations)
- [How to Run the Code](#how-to-run-the-code)
- [PowerPoint Presentation](#powerpoint-presentation)
- [License](#license)

## Overview
In this project, two machine learning models were developed to predict whether an individual is likely to have diabetes or prediabetes based on 21 features.

### Objective:
- Can we predict diabetes risk based on health-related survey data?
- Which factors are most predictive of diabetes?

## Dataset
The dataset used for this project is the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from Kaggle. It contains over 250,000 survey responses from the CDC’s 2015 BRFSS survey.

- **Diabetes_012**: Target variable with 3 classes:
  - 0: No diabetes or diabetes during pregnancy
  - 1: Prediabetes
  - 2: Diabetes

For this project, the target was binarized into two classes:
- **0**: No diabetes
- **1**: Prediabetes or diabetes

### Key Features:
- BMI (Body Mass Index)
- Age
- Income
- Smoking Status
- Physical Activity
- General Health
- High Blood Pressure, Cholesterol, etc.

## Models Used
1. **Random Forest Classifier**:
   - An ensemble method that builds multiple decision trees to make predictions.
   - Provides feature importance to understand which factors have the greatest influence on diabetes risk.
   - Achieved an accuracy of **84.37%** and ROC-AUC of **0.79**.

2. **Logistic Regression**:
   - A simple and interpretable linear model for binary classification.
   - Slightly outperformed Random Forest with an accuracy of **84.85%** and ROC-AUC of **0.82**.

## Results
- **Random Forest**:
  - Accuracy: **84.37%**
  - Precision: **50.26%**
  - ROC-AUC: **0.79**

- **Logistic Regression**:
  - Accuracy: **84.85%**
  - Precision: **54.95%**
  - ROC-AUC: **0.82**

### Confusion Matrix and ROC Curve:
The confusion matrices and ROC curve for both models are shown below:

#### Random Forest
![Confusion Matrix - Random Forest](screenshots/00_confusion_matrix_rf.png)

#### Logistic Regression
![Confusion Matrix - Random Forest](screenshots/00_confusion_matrix_lr.png)

#### ROC Curves
![ROC Curves for both models](screenshots/00_evaluation_roc_curve.png)

## Feature Importance
Random Forest provided insight into the most important features that contribute to the prediction of diabetes:

![Feature Importance - Random Forest](screenshots/00_feature_importances.png)

Top contributing factors:
- **BMI**: Most important factor, directly correlated with diabetes risk.
- **Age**: As expected, the older an individual, the higher the diabetes risk.
- **Income**: Socioeconomic status plays a significant role in diabetes risk, possibly reflecting access to healthcare and lifestyle factors.

## Visualizations
The following visualizations help interpret the models' performance:
- **Confusion Matrix**: Shows the true vs. predicted classifications for both models.
- **ROC Curve**: Evaluates the trade-off between true positive rate and false positive rate for both models.

## How to Run the Code
To run this project on your local machine:

1. Clone the repository:
    ```bash
    git clone https://github.com/swimmingbrain/diabetes-prediction.git
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Python script:
    ```bash
    python project_files/decoding_diabetes_model.py
    ```

4. Check the output for model performance, visualizations, and feature importance.

## PowerPoint Presentation
You can find a PowerPoint presentation for my university conference summarizing this project [here](project_files/decoding_diabetes_conference.pptx).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.