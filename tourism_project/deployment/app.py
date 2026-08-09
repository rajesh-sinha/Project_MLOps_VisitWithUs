import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts the expected **ProdTaken** of a tourism project application
based on its characteristics such as TypeofContact, Occupation, Gender, MaritalStatus, Designation and ProductPitched.
Please enter the app details below to get a Product Taken prediction.
""")

# User input for categorical features
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Super Deluxe', 'Standard', 'King'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced', 'Unmarried'])
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])

# User input for numerical features
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=150.0, value=15.0, step=0.5)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
preferred_property_star = st.number_input("Preferred Property Star (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
number_of_trips = st.number_input("Number of Trips Annually", min_value=1, max_value=50, value=3, step=1)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, step=1)
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=1, step=1)
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Note: LabelEncoder expects numerical values. Since the model expects encoded values
# for categorical features, the Streamlit app should perform the same encoding.
# For simplicity, here we assume the model's preprocessor handles this, but in a real
# deployment, you'd apply the same LabelEncoder transforms that were used during training.
# For this particular model, the preprocessor `make_column_transformer` with `OneHotEncoder`
# will handle the `object` dtypes, so we can pass them as strings.

# Predict button
if st.button("Predict Product Taken"): # Changed button text
    prediction = model.predict(input_data)[0]
    # The model predicts 0 or 1, representing ProdTaken (No/Yes)
    prediction_text = "Yes" if prediction > 0.5 else "No"
    st.subheader("Prediction Result:")
    st.success(f"Predicted Product Taken: **{prediction_text}**")
