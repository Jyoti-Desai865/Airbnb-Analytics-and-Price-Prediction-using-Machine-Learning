# =========================================
# AIRBNB END-TO-END MACHINE LEARNING PIPELINE
# =========================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    r"d:\jyestha project\cleaned_airbnb.csv",
    encoding='latin1',
    low_memory=False
)

print("Dataset Loaded Successfully ✅")

# =========================================
# SELECT FEATURES & TARGET
# =========================================

X = df[['accommodates',
        'bedrooms',
        'review_scores_rating',
        'instant_bookable']]

y = df['price']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# CREATE PIPELINE
# =========================================

pipeline = Pipeline([
    
    # Fill missing values
    ('imputer', SimpleImputer(strategy='median')),
    
    # Scale data
    ('scaler', StandardScaler()),
    
    # Machine Learning Model
    ('model', LinearRegression())
])

# =========================================
# TRAIN PIPELINE
# =========================================

pipeline.fit(X_train, y_train)

print("Pipeline Training Completed ✅")

# =========================================
# MAKE PREDICTIONS
# =========================================

y_pred = pipeline.predict(X_test)

# =========================================
# EVALUATE MODEL
# =========================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------")

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# =========================================
# SAMPLE PREDICTIONS
# =========================================

results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

print("\nSample Predictions")
print(results.head(10))