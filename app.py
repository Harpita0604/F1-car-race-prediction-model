
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained pipeline
pipeline_filename = 'f1_prediction_pipeline.pkl'
try:
    pipeline = joblib.load(pipeline_filename)
    st.success("Prediction pipeline loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Pipeline file '{pipeline_filename}' not found. Please ensure the model training and saving step was completed.")
    pipeline = None # Set pipeline to None if loading fails

st.title('Formula One Race Outcome Prediction')

st.header('Enter Driver and Race Information')

# Input fields for user
driver_name = st.text_input('Driver Name')
team_name = st.text_input('Team Name') # This might need to be mapped to constructorId
standing_position = st.number_input('Current Standing Position', min_value=1)
circuit_name = st.text_input('Circuit Name') # This might need to be mapped to circuitId

# Function to make prediction
def predict_outcome(driver_name, team_name, standing_position, circuit_name, pipeline, combined_df):
    if pipeline is None:
        return "Model not loaded. Cannot make prediction."

    # --- Preprocessing user input to match training data format ---
    # This is a simplified example. In a real application, you would need robust
    # mapping from names to IDs and potentially more feature engineering for the input.

    # Find driverId and constructorId from combined_df based on name (simplified)
    # This requires the combined_df to be available or a mapping file.
    # For this example, let's assume we have a way to get the IDs.
    # In a production app, you'd likely have lookup tables or a more sophisticated approach.

    # Example: Find a driverId (This is highly simplified and might not work for all names)
    driver_info = combined_df[combined_df['driverRef'].str.contains(driver_name, case=False, na=False)].iloc[0] if         combined_df is not None and 'driverRef' in combined_df.columns and combined_df['driverRef'].str.contains(driver_name, case=False, na=False).any() else None

    driver_id = driver_info['driverId'] if driver_info is not None else -1 # Use -1 for unknown

    # Example: Find a constructorId (Simplified)
    constructor_info = combined_df[combined_df['constructorRef'].str.contains(team_name, case=False, na=False)].iloc[0] if         combined_df is not None and 'constructorRef' in combined_df.columns and combined_df['constructorRef'].str.contains(team_name, case=False, na=False).any() else None
    constructor_id = constructor_info['constructorId'] if constructor_info is not None else -1 # Use -1 for unknown

    # Example: Find a circuitId (Simplified)
    circuit_info = combined_df[combined_df['circuitRef'].str.contains(circuit_name, case=False, na=False)].iloc[0] if         combined_df is not None and 'circuitRef' in combined_df.columns and combined_df['circuitRef'].str.contains(circuit_name, case=False, na=False).any() else None
    circuit_id = circuit_info['circuitId'] if circuit_info is not None else -1 # Use -1 for unknown


    # Create a DataFrame with the same columns as the training data's input features
    # This requires knowing the exact feature names and order the pipeline expects.
    # The pipeline's preprocessor expects original categorical and numerical features.
    # The numerical features are: 'age', 'experience_years', 'avg_finishing_position', 'avg_grid_position', 'wins', 'podiums', 'dnfs'
    # The categorical features are: 'driverId', 'constructorId', 'circuitId'

    # For a single prediction, we need to create a DataFrame with one row.
    # We'll need placeholder values for the engineered features for the input.
    # A robust solution would pre-calculate or estimate these based on the input driver/team.
    # For this example, let's use mean values from the training data (this is a simplification).

    # We need to access the preprocessor in the pipeline to get the feature names it expects.
    # However, directly accessing feature names after ColumnTransformer is complex.
    # A simpler approach for prediction is to create a DataFrame with the original raw features
    # that the ColumnTransformer was fitted on.

    # Create a dictionary with the raw input features that the pipeline's preprocessor expects
    # We need to provide values for all features the pipeline was trained on, even if they are not directly
    # user inputs. For the engineered features ('age', 'experience_years', etc.), we'll use placeholder values.
    # In a real application, you'd calculate these based on the input driver and the historical data.

    # Let's create a DataFrame with the raw features that the pipeline expects.
    # The pipeline expects columns: 'age', 'experience_years', 'avg_finishing_position', 'avg_grid_position', 'wins', 'podiums', 'dnfs', 'driverId', 'constructorId', 'circuitId'

    # We need to get the average values of the numerical features from the training data
    # to use as placeholders for the input. We can get this from the trained pipeline's preprocessor.
    # Accessing fitted transformers to get means can be tricky.
    # Let's use the means calculated earlier during data preparation (assuming combined_df is available).
    # This is another simplification.

    # Get the mean of numerical features from the cleaned training data (df_cleaned)
    # This requires df_cleaned to be available, which might not be the case in the deployed app.
    # A better approach is to save the mean values along with the pipeline or calculate them from combined_df.
    # Let's calculate them from the combined_df for this example, handling potential NaNs.

    numerical_features = [
        'age',
        'experience_years',
        'avg_finishing_position',
        'avg_grid_position',
        'wins',
        'podiums',
        'dnfs'
    ]
    numerical_means = combined_df[numerical_features].mean()


    input_data = {
        'age': numerical_means.get('age', 0), # Use .get to handle cases where feature might be missing
        'experience_years': numerical_means.get('experience_years', 0),
        'avg_finishing_position': numerical_means.get('avg_finishing_position', combined_df['positionText_numeric'].mean() if 'positionText_numeric' in combined_df.columns else 10), # Use mean of target as a fallback
        'avg_grid_position': numerical_means.get('avg_grid_position', combined_df['grid_numeric'].mean() if 'grid_numeric' in combined_df.columns else 10), # Use mean of grid as a fallback
        'wins': numerical_means.get('wins', 0),
        'podiums': numerical_means.get('podiums', 0),
        'dnfs': numerical_means.get('dnfs', 0),
        'driverId': driver_id,
        'constructorId': constructor_id,
        'circuitId': circuit_id,
    }

    # Create a DataFrame with a single row for prediction
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns in input_df matches the order the pipeline expects
    # The pipeline's preprocessor was fitted on X_train, which had columns in the order:
    # numerical_features + categorical_features
    # We need to make sure the categorical features are also defined here
    categorical_features = ['driverId', 'constructorId', 'circuitId'] # Need to explicitly define them here as well
    expected_columns = numerical_features + categorical_features

    # Reindex input_df to match the expected column order, filling missing columns with default values if necessary
    input_df = input_df.reindex(columns=expected_columns, fill_value=0) # Fill with 0 or another appropriate default


    # Make prediction
    try:
        prediction = pipeline.predict(input_df)
        return f"Predicted Finishing Position: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error during prediction: {e}"


# Add a button to trigger prediction
if st.button('Predict Outcome'):
    # combined_df needs to be available to map names to IDs.
    # In a real deployed app, you would load the necessary mapping data or
    # pre-calculate and store the IDs. For this Colab environment, we assume
    # combined_df is available from previous cells.
    if 'combined_df' in locals():
         prediction_result = predict_outcome(driver_name, team_name, standing_position, circuit_name, pipeline, combined_df)
         st.write(prediction_result)
    else:
        st.error("Error: 'combined_df' not found. Cannot map driver, team, or circuit names to IDs.")

