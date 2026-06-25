import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("sales_data.csv")

# Display first rows
print("Dataset:")
print(data.head())

# Features and target
X = data[['Month']]
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Predict future sales
future_months = pd.DataFrame({
    'Month': [21, 22, 23, 24, 25]
})

future_predictions = model.predict(future_months)

print("\nFuture Predictions:")
for month, prediction in zip(future_months['Month'], future_predictions):
    print(f"Month {month}: Predicted Sales = {prediction:.2f}")

# Visualization
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.scatter(
    future_months,
    future_predictions,
    color='green',
    label='Future Predictions'
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Prediction Using Linear Regression")
plt.legend()

plt.show()