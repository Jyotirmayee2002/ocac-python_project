import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Title of the app
st.title("Linear Regression Model Trainer and Predictor")

# Input data for training
st.write("Enter training data")
X_train_input = st.text_area("Enter the features (X) separated by commas, one sample per line", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n2, 3, 4, 5, 6, 7, 8, 9, 10, 11\n3, 4, 5, 6, 7, 8, 9, 10, 11, 12")
y_train_input = st.text_area("Enter the target values (y) separated by commas, one per line", "2\n3\n4")

# Convert input data to appropriate format
X_train = np.array([list(map(float, x.split(','))) for x in X_train_input.strip().split('\n')])
y_train = np.array(list(map(float, y_train_input.strip().split('\n'))))

# Define the preprocessor
preprocessor = StandardScaler()

# Fit the preprocessor
preprocessor.fit(X_train)

# Transform the training data
X_train_transformed = preprocessor.transform(X_train)

# Define the linear regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train_transformed, y_train)

# Save the preprocessor and model to files
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(lr_model, 'linear_regression_model.pkl')
st.write("Model and preprocessor saved as 'linear_regression_model.pkl' and 'preprocessor.pkl'")

# Input data for prediction
st.write("Enter data for prediction")
state = st.text_input("State")
inc_2000_01 = st.number_input("Income 2000-01", value=0.0)
inc_2011_12 = st.number_input("Income 2011-12", value=0.0)
lit_2001 = st.number_input("Literacy 2001", value=0.0)
lit_2011 = st.number_input("Literacy 2011", value=0.0)
pop_2001 = st.number_input("Population 2001", value=0.0)
pop_2011 = st.number_input("Population 2011", value=0.0)
sex_ratio_2001 = st.number_input("Sex Ratio 2001", value=0.0)
sex_ratio_2011 = st.number_input("Sex Ratio 2011", value=0.0)
unemploy_2001 = st.number_input("Unemployment 2001", value=0.0)
poverty_2011 = st.number_input("Poverty 2011", value=0.0)

def predict_unemployment(state, inc_2000_01, inc_2011_12, lit_2001, lit_2011, pop_2001, pop_2011, sex_ratio_2001, sex_ratio_2011, unemploy_2001, poverty_2011):
    features = np.array([[inc_2000_01, inc_2011_12, lit_2001, lit_2011, pop_2001, pop_2011, sex_ratio_2001, sex_ratio_2011, unemploy_2001, poverty_2011]])
    
    # Load the preprocessor and model
    preprocessor = joblib.load('preprocessor.pkl')
    lr_model = joblib.load('linear_regression_model.pkl')
    
    # Transform the features
    processed_features = preprocessor.transform(features)
    
    # Predict the unemployment rate
    prediction = lr_model.predict(processed_features)
    return prediction[0]

if st.button("Predict Unemployment Rate"):
    prediction = predict_unemployment(state, inc_2000_01, inc_2011_12, lit_2001, lit_2011, pop_2001, pop_2011, sex_ratio_2001, sex_ratio_2011, unemploy_2001, poverty_2011)
    st.write(f"The predicted unemployment rate is: {prediction}")