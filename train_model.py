import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('StudentPerformanceFactors.csv')

df = data.copy()
target = 'Exam_Score'
encode = ['Parental_Involvement','Access_to_Resources',
          'Extracurricular_Activities','Motivation_Level',
          'Internet_Access','Family_Income','Teacher_Quality',
          'School_Type','Peer_Influence','Learning_Disabilities',
          'Gender', 'Distance_from_Home','Parental_Education_Level']

# One-hot encode categorical columns and track columns
encoded_columns = []

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    df.drop(columns=[col], inplace=True)
    encoded_columns.extend(dummy.columns)

# Ensure the target is in the dataset.
if target not in df.columns:
    raise ValueError(f"The target column '{target}' is not present in the dataset.")

# Get the column names as a list
column_names = df.columns.tolist()

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(target, axis=1))

# Prepare the feature matrix (X) and target variable (y)
X = scaled_features
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the scaler and model using pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('exam_score_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the encoded feature columns for future use
with open('encoded_columns.pkl', 'wb') as columns_file:
    pickle.dump(encoded_columns, columns_file)

# Save all feature columns for future use
with open('all_columns.pkl', 'wb') as all_columns_file:
    pickle.dump(column_names, all_columns_file)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model training complete.")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
