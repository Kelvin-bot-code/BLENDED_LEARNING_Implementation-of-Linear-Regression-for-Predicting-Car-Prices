# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset: Load the dataset containing car prices and relevant features.
3. Data Preprocessing: Handle missing values and perform feature selection if necessary.
4. Split Data: Split the dataset into training and testing sets.
5. Train Model: Create a linear regression model and fit it to the training data.
6. Make Predictions: Use the model to make predictions on the test set.
7. Evaluate Model: Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions: Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results: Display the predictions and evaluation metrics.

## Program:
```python
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
register no: 212224040157
name       : kelvin k
*/
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")
print("="*50)

print("\nMODEL PERFORMANCE:")
mse=mean_squared_error(y_test, y_pred)
print('Mean squared error=',mse)
rmse=np.sqrt(mse)
print('Root mean squared error=',rmse)
r2score=r2_score(y_test, y_pred)
print('R-squared=',r2score)

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic:{dw_test:.2f}","\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
![Screenshot 2025-04-27 142208](https://github.com/user-attachments/assets/680f1ebd-83b5-4ca7-acf9-93c0cefc77d8)

*1. Linearity check*

![Screenshot 2025-04-27 142150](https://github.com/user-attachments/assets/8215d21a-ba9c-45a6-bd09-9a0525bb850e)

*2. Independence (Durbin-Watson)*

![Screenshot 2025-04-27 142200](https://github.com/user-attachments/assets/306f11fa-d06d-4733-a525-7d6d6e4780e9)

*3. Homoscedasticity*

![Screenshot 2025-04-27 142140](https://github.com/user-attachments/assets/aacd48b7-566a-4b5f-8767-bbf50fc00dc6)

*4. Normality of residuals*

![Screenshot 2025-04-27 142129](https://github.com/user-attachments/assets/0f7ffe2f-247f-4404-9db6-f7ab087d864d)

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
