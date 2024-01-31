from sklearn.linear_model import LinearRegression

# Sample data
data = [
    {'CPI': 5000, 'Discounts': 3, 'Offers': 20, 'Sales': None},  # Target Sales for prediction
    {'CPI': 4000, 'Discounts': 8, 'Offers': 19, 'Sales': None}   # Target Sales for prediction
]

# Training data
X_train = [
    [5000, 3, 20],  # CPI, Discounts, Offers
    [4000, 8, 19]   # CPI, Discounts, Offers
]

y_train = [5000, 4000]  # Actual Sales

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting sales for new data
X_new = [
    [5000, 3, 20],  # CPI, Discounts, Offers
    [4000, 8, 19]   # CPI, Discounts, Offers
]

predicted_sales = model.predict(X_new)

# Print predicted sales
for i, data_point in enumerate(data):
    data_point['Sales'] = predicted_sales[i]
    print(f"Predicted Sales for CPI={data_point['CPI']}, Discounts={data_point['Discounts']}%, Offers={data_point['Offers']} : {data_point['Sales']}")
