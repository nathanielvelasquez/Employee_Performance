import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('/Users/nate/Desktop/Projects/Python/Employee_Performance/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the first few rows of the dataset
print(data.head())

# Explore data statistics
print(data.describe())

# Handle missing values if any
data = data.dropna()

# Encodes categorical variables
selected_features = ['Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance']

# Split the data into features (X) and target variable (y)
X = data[selected_features]
y = data['PerformanceRating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Trains a Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Feature Importance Analysis
feature_importances = model.feature_importances_

##Create a DataFrame to visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Employee Performance')
plt.show()

# My analysis has found the top three determinates for employee performance scores to be:
# 1) Environment Satisfaction
# 2) Education
# 3) Relationship Satisfaction