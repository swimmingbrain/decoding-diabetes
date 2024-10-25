import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# split dataset into features and target
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# binarize target variable: 0 - no diabetes, 1 - prediabetes or diabetes
y_binary = y.apply(lambda x: 1 if x > 0 else 0)

# apply undersampling to balance the classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y_binary)

# calculate control-to-treatment ratio
control_count = sum(y_resampled == 0)
treatment_count = sum(y_resampled == 1)

control_to_treatment_ratio = control_count / treatment_count
print(f"Control-to-Treatment Ratio: {control_to_treatment_ratio}")

# reset the index for both X_resampled and y_resampled to align indices
X_resampled = X_resampled.reset_index(drop=True)
y_resampled = pd.Series(y_resampled).reset_index(drop=True)

# standardize the X variables for univariate logistic regression
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X_resampled.columns)

# store the results from univariate logistic regressions
logit_results = {}

# perform univariate logistic regression for each feature
for column in X_scaled.columns:
    X_univariate = sm.add_constant(X_scaled[[column]])  # add constant for intercept
    model = sm.Logit(y_resampled, X_univariate).fit(disp=0)
    logit_results[column] = {
        'coef': model.params[1],
        'p-value': model.pvalues[1],
        'odds_ratio': np.exp(model.params[1])
    }

# convert results to a DataFrame for better readability
univariate_results = pd.DataFrame(logit_results).T.sort_values('p-value')
print("Top features from univariate analysis:")
print(univariate_results.head())

# feature selection based on p-values (p < 0.05)
significant_features = univariate_results[univariate_results['p-value'] < 0.05].index
print(f"Significant features selected: {significant_features}")

# use only significant features for the models
X_significant = X_scaled[significant_features]

# define cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Classifier with cross-validation predictions
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_rf_proba = cross_val_predict(rf_model, X_significant, y_resampled, cv=cv, method='predict_proba')[:, 1]
y_pred_rf = (y_pred_rf_proba >= 0.5).astype(int)

# Logistic Regression Classifier with c-v
log_reg = LogisticRegression(max_iter=1000)
y_pred_lr_proba = cross_val_predict(log_reg, X_significant, y_resampled, cv=cv, method='predict_proba')[:, 1]
y_pred_lr = (y_pred_lr_proba >= 0.5).astype(int)

# Gradient Boosting Classifier with c-v
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
y_pred_gb_proba = cross_val_predict(gb_model, X_significant, y_resampled, cv=cv, method='predict_proba')[:, 1]
y_pred_gb = (y_pred_gb_proba >= 0.5).astype(int)

# compute metrics for Random Forest
accuracy_rf = accuracy_score(y_resampled, y_pred_rf)
precision_rf = precision_score(y_resampled, y_pred_rf)
roc_auc_rf = roc_auc_score(y_resampled, y_pred_rf_proba)

# compute metrics for Logistic Regression
accuracy_lr = accuracy_score(y_resampled, y_pred_lr)
precision_lr = precision_score(y_resampled, y_pred_lr)
roc_auc_lr = roc_auc_score(y_resampled, y_pred_lr_proba)

# compute metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_resampled, y_pred_gb)
precision_gb = precision_score(y_resampled, y_pred_gb)
roc_auc_gb = roc_auc_score(y_resampled, y_pred_gb_proba)

# print metrics for each model
print("\nModel Evaluation Metrics:")

print("\nRandom Forest:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"ROC-AUC: {roc_auc_rf:.4f}")

print("\nLogistic Regression:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"ROC-AUC: {roc_auc_lr:.4f}")

print("\nGradient Boosting:")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"ROC-AUC: {roc_auc_gb:.4f}")

# confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_resampled, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Reds")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# confusion matrix for Logistic Regression
conf_matrix_lr = confusion_matrix(y_resampled, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# confusion matrix for Gradient Boosting
conf_matrix_gb = confusion_matrix(y_resampled, y_pred_gb)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_gb, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# calculate ROC curve for each model
fpr_rf, tpr_rf, _ = roc_curve(y_resampled, y_pred_rf_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_resampled, y_pred_lr_proba)
fpr_gb, tpr_gb, _ = roc_curve(y_resampled, y_pred_gb_proba)

# plot ROC Curves
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting (ROC-AUC = %0.2f)" % roc_auc_gb)
plt.plot(fpr_rf, tpr_rf, label="Random Forest (ROC-AUC = %0.2f)" % roc_auc_rf)
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (ROC-AUC = %0.2f)" % roc_auc_lr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
