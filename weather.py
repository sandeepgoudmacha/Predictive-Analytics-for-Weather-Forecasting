import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("🌤️ Weather Forecasting using Linear Regression & Classification")

# Upload weather data CSV
uploaded_file = st.file_uploader("Upload your weather data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Fallback to a sample dataset if none is uploaded
    st.error("Please upload a CSV file with weather data.")
    st.stop()

# Displaying dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Convert 'Formatted Date' to datetime
if 'Formatted Date' in df.columns:
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce')

# Handle missing values by dropping rows with NaN in any column
df = df.dropna()

# Check if expected columns exist before proceeding
required_columns = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Apparent Temperature (C)', 'Temperature (C)', 'Precip Type', 'Summary']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Column '{col}' is missing from the dataset. Please check the CSV file.")
        st.stop()

# Feature selection for regression (example: predicting 'Temperature (C)')
features = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Apparent Temperature (C)']
target = 'Temperature (C)'

# Encoding categorical features (e.g., 'Precip Type' and 'Summary')
label_encoder = LabelEncoder()
df['Precip Type'] = label_encoder.fit_transform(df['Precip Type'])
df['Summary'] = label_encoder.fit_transform(df['Summary'])

# Selecting features and target for regression (Temperature)
X = df[features]
y = df[target]

# Feature Scaling (Standardizing the features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets for temperature prediction
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Linear Regression model for Temperature
model_temp = LinearRegression()
model_temp.fit(X_train, y_train)

# Make temperature predictions
y_pred_temp = model_temp.predict(X_test)

# Display metrics for temperature prediction
st.subheader("📊 Temperature Model Evaluation")
mae_temp = mean_absolute_error(y_test, y_pred_temp)
mse_temp = mean_squared_error(y_test, y_pred_temp)
rmse_temp = np.sqrt(mse_temp)
r2_temp = r2_score(y_test, y_pred_temp)

st.write(f"Mean Absolute Error (MAE): {mae_temp:.2f}")
st.write(f"Mean Squared Error (MSE): {mse_temp:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_temp:.2f}")
st.write(f"R-squared (R²): {r2_temp:.2f}")

# Plotting the results for temperature
st.subheader("📈 Temperature Predictions vs Actual")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(y_test.values, label='Actual')
ax.plot(y_pred_temp, label='Predicted')
ax.set_title("Weather Forecasting: Temperature Predictions vs Actual")
ax.set_xlabel("Samples")
ax.set_ylabel(f"{target} Value")
ax.legend()
st.pyplot(fig)

# Feature selection for classification (predicting weather type)
features_class = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Apparent Temperature (C)']
target_class = 'Summary'

# Selecting features and target for classification (Weather Type)
X_class = df[features_class]
y_class = df[target_class]

# Split the data into training and testing sets for weather type prediction
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# Train the Logistic Regression model for Weather Type (classification)
model_weather = LogisticRegression(max_iter=200)
model_weather.fit(X_train_class, y_train_class)

# Make weather type predictions
y_pred_class = model_weather.predict(X_test_class)

# Display metrics for weather type prediction
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_class, y_pred_class)

st.subheader("📊 Weather Type Model Evaluation")
st.write(f"Accuracy of Weather Type Prediction: {accuracy*100:.2f}%")

# Plotting the results for weather type prediction (confusion matrix)
st.subheader("📊 Weather Type Prediction Confusion Matrix")
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_class, y_pred_class)
fig_cm, ax_cm = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
ax_cm.set_title("Confusion Matrix for Weather Type Prediction")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# Predict on new data
st.subheader("🔮 Predict Future Weather")

# Take user input for future prediction
new_data = {}
for feature in features:
    new_data[feature] = st.number_input(f"Enter {feature}", min_value=0, value=0)

# Convert input into dataframe for scaling and prediction
input_data = pd.DataFrame([new_data])
input_data_scaled = scaler.transform(input_data)

# Make predictions based on input
if st.button("Predict Weather"):
    # Predict temperature
    temp_prediction = model_temp.predict(input_data_scaled)
    
    # Predict weather type
    weather_prediction = model_weather.predict(input_data_scaled)
    
    # Decode the weather type back to original string
    weather_type = label_encoder.inverse_transform(weather_prediction)[0]
    
    st.write(f"Predicted Temperature (C): {temp_prediction[0]:.2f}")
    st.write(f"Predicted Weather Type: {weather_type}")
