import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

st.title("F1 2025 Qualifying Prediction Model")

# -------------------------
# Functions
# -------------------------
def fetch_f1_data(year, round_number):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver', 'TeamName': 'Team'})

        # Weather columns
        results['AirTemp'] = None
        results['TrackTemp'] = None
        results['Humidity'] = None
        results['WindSpeed'] = None
        results['Weather'] = None

        weather_data = quali.weather_data
        if weather_data is not None and not weather_data.empty:
            first_weather_sample = weather_data.iloc[0]
            results['AirTemp'] = first_weather_sample.get('AirTemp', None)
            results['TrackTemp'] = first_weather_sample.get('TrackTemp', None)
            results['Humidity'] = first_weather_sample.get('Humidity', None)
            results['WindSpeed'] = first_weather_sample.get('WindSpeed', None)
            results['Weather'] = first_weather_sample.get('Weather', None)

        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)

        return results
    except Exception as e:
        st.warning(f"Error fetching data for {year} round {round_number}: {e}")
        return None

def calculate_driver_form(df, window_size=3):
    df_sorted = df.sort_values(by=['Driver', 'Year', 'Round'])
    df_sorted['Driver_Form'] = df_sorted.groupby('Driver')['Q3_sec'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df_sorted['Driver_Form'] = df_sorted.groupby('Driver')['Driver_Form'].shift(1)
    return df_sorted

def clean_data(df):
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fetch_recent_data():
    all_data = []
    year_to_fetch = 2024
    for round_num in range(1, 6):
        st.info(f"Fetching data for {year_to_fetch} round {round_num}...")
        df = fetch_f1_data(year_to_fetch, round_num)
        if df is not None:
            df['Year'] = year_to_fetch
            df['Round'] = round_num
            all_data.append(df)
    return all_data

def apply_performance_factors(predictions_df):
    team_adjustments = {
        'Red Bull Racing': -0.3, 'Ferrari': -0.2, 'McLaren': -0.15, 'Mercedes': -0.15,
        'Aston Martin': 0.1, 'RB': 0.2, 'Williams': 0.3, 'Haas F1 Team': 0.4,
        'Kick Sauber': 0.4, 'Alpine': 0.5
    }
    driver_adjustments = {
        'Max Verstappen': -0.2, 'Charles Leclerc': -0.1, 'Carlos Sainz': -0.1,
        'Lando Norris': -0.1, 'Oscar Piastri': 0.0, 'Sergio Perez': 0.0,
        'Lewis Hamilton': 0.0, 'George Russell': 0.0, 'Fernando Alonso': 0.0,
        'Lance Stroll': 0.1, 'Alexander Albon': 0.1, 'Daniel Ricciardo': 0.1,
        'Yuki Tsunoda': 0.2, 'Valtteri Bottas': 0.2, 'Zhou Guanyu': 0.3,
        'Kevin Magnussen': 0.3, 'Nico Hulkenberg': 0.3, 'Logan Sargeant': 0.4,
        'Pierre Gasly': 0.4, 'Esteban Ocon': 0.4
    }
    for idx, row in predictions_df.iterrows():
        adj = team_adjustments.get(row['Team'], 0.5) + driver_adjustments.get(row['Driver'], 0.2)
        predictions_df.loc[idx, 'Predicted_Q3'] = row['Predicted_Q3_Model'] + adj + np.random.uniform(-0.05,0.05)
    return predictions_df

# -------------------------
# Fetch and prepare data
# -------------------------
all_data = fetch_recent_data()
if not all_data:
    st.error("Failed to fetch F1 data")
    st.stop()

combined_df = pd.concat(all_data, ignore_index=True)
combined_df = calculate_driver_form(combined_df)
combined_df = clean_data(combined_df)

features = ['Q1_sec','Q2_sec','AirTemp','TrackTemp','Humidity','WindSpeed','Driver_Form']
target = 'Q3_sec'

valid_data = combined_df.dropna(subset=[target])
X = valid_data[features]
y = valid_data[target]

imputer = SimpleImputer(strategy='median')
X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# -------------------------
# Train models
# -------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

models = {
    "Linear Regression": lr_model,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

st.subheader("Model Evaluation on Test Data")
for name, m in models.items():
    y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"{name} - MAE: {mae:.2f}s, R2: {r2:.2f}")

# -------------------------
# Japanese GP 2025 Predictions
# -------------------------
driver_teams_2025 = {
    'Max Verstappen':'Red Bull Racing','Sergio Perez':'Red Bull Racing','Charles Leclerc':'Ferrari',
    'Carlos Sainz':'Ferrari','Lewis Hamilton':'Mercedes','George Russell':'Mercedes','Lando Norris':'McLaren',
    'Oscar Piastri':'McLaren','Fernando Alonso':'Aston Martin','Lance Stroll':'Aston Martin',
    'Daniel Ricciardo':'RB','Yuki Tsunoda':'RB','Alexander Albon':'Williams','Logan Sargeant':'Williams',
    'Valtteri Bottas':'Kick Sauber','Zhou Guanyu':'Kick Sauber','Kevin Magnussen':'Haas F1 Team',
    'Nico Hulkenberg':'Haas F1 Team','Pierre Gasly':'Alpine','Esteban Ocon':'Alpine'
}

jgp_df = pd.DataFrame(list(driver_teams_2025.items()), columns=['Driver','Team'])

# Weather inputs in sidebar
st.sidebar.header("Weather Inputs")
jgp_df['AirTemp'] = st.sidebar.number_input("Air Temperature", value=combined_df['AirTemp'].mean())
jgp_df['TrackTemp'] = st.sidebar.number_input("Track Temperature", value=combined_df['TrackTemp'].mean())
jgp_df['Humidity'] = st.sidebar.number_input("Humidity", value=combined_df['Humidity'].mean())
jgp_df['WindSpeed'] = st.sidebar.number_input("Wind Speed", value=combined_df['WindSpeed'].mean())

driver_avg_q3 = combined_df.groupby('Driver')['Q3_sec'].mean().to_dict()
jgp_df['Driver_Form'] = jgp_df['Driver'].map(driver_avg_q3).fillna(combined_df['Q3_sec'].mean())

driver_avg_q1 = combined_df.groupby('Driver')['Q1_sec'].mean().to_dict()
driver_avg_q2 = combined_df.groupby('Driver')['Q2_sec'].mean().to_dict()
jgp_df['Q1_sec'] = jgp_df['Driver'].map(driver_avg_q1).fillna(combined_df['Q1_sec'].mean())
jgp_df['Q2_sec'] = jgp_df['Driver'].map(driver_avg_q2).fillna(combined_df['Q2_sec'].mean())

X_predict = jgp_df[features]
X_predict_clean = pd.DataFrame(imputer.transform(X_predict), columns=features)

chosen_model = gb_model
jgp_df['Predicted_Q3_Model'] = chosen_model.predict(X_predict_clean)
jgp_df = apply_performance_factors(jgp_df)

results_df = jgp_df[['Driver','Team','Predicted_Q3']].sort_values('Predicted_Q3').reset_index(drop=True)
results_df.index += 1

st.subheader("Japanese GP 2025 Qualifying Predictions (Adjusted)")
st.dataframe(results_df)
