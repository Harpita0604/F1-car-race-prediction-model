import streamlit as st
import pandas as pd
import joblib

# Load the trained model and prepared data
# Assuming the trained model was saved as 'f1_model.pkl' and data as 'prepared_f1_data.csv'
try:
    model = joblib.load('f1_model.pkl')
    prepared_df = pd.read_csv('prepared_f1_data.csv')
except FileNotFoundError:
    st.error("Error: Model file (f1_model.pkl) or data file (prepared_f1_data.csv) not found.")
    st.stop() # Stop the app if files are not found

# Set up the Streamlit page configuration
st.set_page_config(page_title="F1 Race Result Predictor", layout="wide")

st.title("Formula 1 Race Result Predictor")

st.write("Enter the details below to predict the race points.")

# Get unique driver names and constructor names from the prepared data
# Combine forename and surname for full driver name
prepared_df['full_name'] = prepared_df['forename'] + ' ' + prepared_df['surname']
unique_drivers = sorted(prepared_df['full_name'].unique())
unique_constructors = sorted(prepared_df['name_constructor'].unique())

# Create input widgets
selected_constructor = st.selectbox("Select Car Brand (Constructor):", unique_constructors)
driver_standing_position = st.number_input("Enter Driver Standing Position:", min_value=0, value=10, step=1)
selected_driver = st.selectbox("Select Driver Name:", unique_drivers)


# Add a button to trigger the prediction (prediction logic will be added in the next step)
if st.button("Predict Points"):
    st.write("Prediction will be shown here after implementing the prediction logic.")

# Add the prediction logic
if st.button("Predict Points"):
    # Create a DataFrame with the input values, matching the structure of X_train
    # Need to handle one-hot encoding for the selected constructor
    input_data = {
        'year': [2024], # Assuming a prediction for the current/upcoming year
        'round': [1], # Assuming prediction for the first round, this could be another input
        'position_standing': [driver_standing_position],
        'points_standing': [0], # Starting points for a new race/season, could be an input
        'wins': [0], # Starting wins for a new race/season, could be an input
        'driver_avg_points_last_3': [0], # Need to calculate/estimate based on input driver
        'constructor_avg_points_last_3': [0], # Need to calculate/estimate based on input constructor
        'driver_constructor_interaction': [0], # Interaction based on estimated averages
        # Add columns for all possible one-hot encoded constructors, set to False initially
    }

    # Add one-hot encoded columns for constructors
    for constructor in unique_constructors:
        input_data[f'name_constructor_{constructor}'] = [False]

    # Set the selected constructor's one-hot encoded column to True
    if f'name_constructor_{selected_constructor}' in input_data:
        input_data[f'name_constructor_{selected_constructor}'][0] = True
    else:
        st.error(f"Selected constructor '{selected_constructor}' not found in training data.")
        st.stop()

    # Create the input DataFrame
    input_df = pd.DataFrame(input_data)

    # Ensure the order of columns matches the training data
    # This is crucial for consistent predictions
    # We need the columns from X_train to ensure the input_df has the same structure
    train_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=train_cols, fill_value=0)


    # Make the prediction
    predicted_points = model.predict(input_df)

    # Display the prediction
    st.subheader("Predicted Race Points:")
    st.write(f"{predicted_points[0]:.2f}")
