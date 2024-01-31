import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Cybill_Score': [700, 600, 750, 650, 720],
    'Age': [35, 45, 30, 50, 40],
    'Insurance': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Debit_Card': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Cards': [2, 1, 3, 2, 2],
    'Loan_Eligibility': ['Yes', 'No', 'Yes', 'No', 'Yes']  # Target variable
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical
df['Insurance'] = df['Insurance'].map({'Yes': 1, 'No': 0})
df['Debit_Card'] = df['Debit_Card'].map({'Yes': 1, 'No': 0})

# Features and target variable
X = df[['Cybill_Score', 'Age', 'Insurance', 'Debit_Card', 'Cards']]
y = df['Loan_Eligibility']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example prediction
new_customer = pd.DataFrame({
    'Cybill_Score': [680],
    'Age': [25],
    'Insurance': [1],
    'Debit_Card': [1],
    'Cards': [2]
})

# Scale features
new_customer_scaled = scaler.transform(new_customer)

# Predict loan eligibility
prediction = model.predict(new_customer_scaled)
print(f"Predicted Loan Eligibility: {prediction[0]}")
