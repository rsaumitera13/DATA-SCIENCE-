# MULTIPLE LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# 1. Load Data & EDA


file_path = 'ToyotaCorolla - MLR.csv' 
try:
    df = pd.read_csv(r"C:\Users\saumitra sundar rath\Downloads\Multiple Linear Regression (1)\ToyotaCorolla - MLR.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please check the file path.")
    exit()

# Descriptive Statistics
print("\n--- Data Information ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# Data Cleaning
# Dropping 'Cylinders' because it has constant variance (all values are 4)
if 'Cylinders' in df.columns and df['Cylinders'].nunique() <= 1:
    print("\nDropping 'Cylinders' column (constant value).")
    df = df.drop(columns=['Cylinders'])

# Visualization: Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# 2. Preprocessing

# One-Hot Encoding for Categorical Variable 'Fuel_Type'
df_processed = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=True)

# Define Features (X) and Target (y)
target_col = 'Price'
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Model Building


# --- Model 1: All Features (OLS) ---
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

# --- Model 2: Selected Features ---
# Using only strong predictors found in EDA
features_m2 = ['Age_08_04', 'KM', 'Weight', 'HP']
# Filtering columns to ensure they exist
features_m2 = [c for c in features_m2 if c in X_train.columns]

X_train_m2 = X_train[features_m2]
X_test_m2 = X_test[features_m2]

model2 = LinearRegression()
model2.fit(X_train_m2, y_train)
y_pred2 = model2.predict(X_test_m2)

# --- Model 3: Standardized Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model3 = LinearRegression()
model3.fit(X_train_scaled, y_train)
y_pred3 = model3.predict(X_test_scaled)


# 4. Evaluation

def print_metrics(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {model_name} ---")
    print(f"R-Squared: {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")

print_metrics(y_test, y_pred1, "Model 1 (All Features)")
print_metrics(y_test, y_pred2, "Model 2 (Selected Features)")
print_metrics(y_test, y_pred3, "Model 3 (Standardized)")

# Interpret Coefficients (from Model 1)
print("\n--- Coefficients (Model 1) ---")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model1.coef_})
print(coef_df.sort_values(by='Coefficient', ascending=False))


# 5. Lasso & Ridge Regression

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print_metrics(y_test, y_pred_lasso, "Lasso Regression")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print_metrics(y_test, y_pred_ridge, "Ridge Regression")


--- Data Information ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1436 entries, 0 to 1435
Data columns (total 11 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   Price      1436 non-null   int64 
 1   Age_08_04  1436 non-null   int64 
 2   KM         1436 non-null   int64 
 3   Fuel_Type  1436 non-null   object
 4   HP         1436 non-null   int64 
 5   Automatic  1436 non-null   int64 
 6   cc         1436 non-null   int64 
 7   Doors      1436 non-null   int64 
 8   Cylinders  1436 non-null   int64 
 9   Gears      1436 non-null   int64 
 10  Weight     1436 non-null   int64 
dtypes: int64(10), object(1)
memory usage: 123.5+ KB
None

--- Summary Statistics ---
              Price    Age_08_04             KM           HP    Automatic  \
...
75%     1600.00000     5.000000        4.0     5.000000  1085.00000  
max    16000.00000     5.000000        4.0     6.000000  1615.00000  

Dropping 'Cylinders' column (constant value).

-- Model 1 (All Features) ---
R-Squared: 0.8349
RMSE:      1484.2654

--- Model 2 (Selected Features) ---
R-Squared: 0.8506
RMSE:      1411.8502

--- Model 3 (Standardized) ---
R-Squared: 0.8349
RMSE:      1484.2654

--- Coefficients (Model 1) ---
            Feature  Coefficient
9  Fuel_Type_Petrol  1370.808910
6             Gears   551.600710
3         Automatic   148.830927
7            Weight    25.884958
2                HP    14.039479
1                KM    -0.016231
4                cc    -0.030372
5             Doors   -60.310974
8  Fuel_Type_Diesel   -68.548757
0         Age_08_04  -120.830458
...

--- Ridge Regression ---
R-Squared: 0.8350
RMSE:      1483.5575


