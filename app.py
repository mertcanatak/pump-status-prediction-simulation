import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time
from keras.models import load_model
from modules.database import insert_data, load_data, save_predictions, get_last_id
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'data', 'lstm_model.keras')
csv_file_path = os.path.join(BASE_DIR, 'data', 'updated_usablesensors.csv')

# LSTM Model and data
model = load_model(model_path)
data = pd.read_csv(csv_file_path)


st.title('Gerçek Zamanlı Veri Görselleştirme ve Kaydetme')

chart_placeholder = st.empty()

if 'current_id' not in st.session_state:
    last_id = get_last_id()
    if last_id:
        st.session_state.current_id = last_id + 1
    else:
        st.session_state.current_id = 1


def calculate_stats(data):
    stats = {}
    for column in data.columns:
        if column not in ['id', 'datetime']:
            stats[column] = {
                'mean': data[column].astype(float).mean(),
                'std': data[column].astype(float).std()
            }
    return stats

def generate_synthetic_data(id_value, stats):
    synthetic_data = {
        "id": str(id_value),
        "datetime": datetime.now().isoformat(),
        "Motor Casing Vibration": str(
            np.round(np.random.normal(stats['Motor Casing Vibration']['mean'], stats['Motor Casing Vibration']['std']), 2)),
        "Motor Frequency A": str(
            np.round(np.random.normal(stats['Motor Frequency A']['mean'], stats['Motor Frequency A']['std']), 2)),
        "Motor Frequency B": str(
            np.round(np.random.normal(stats['Motor Frequency B']['mean'], stats['Motor Frequency B']['std']), 2)),
        "Motor Frequency C": str(
            np.round(np.random.normal(stats['Motor Frequency C']['mean'], stats['Motor Frequency C']['std']), 2)),
        "Motor Speed": str(np.round(np.random.normal(stats['Motor Speed']['mean'], stats['Motor Speed']['std']), 2)),
        "Motor Current": str(np.round(np.random.normal(stats['Motor Current']['mean'], stats['Motor Current']['std']), 2)),
        "Motor Active Power": str(np.round(np.random.normal(stats['Motor Active Power']['mean'], stats['Motor Active Power']['std']), 2)),
        "Motor Apparent Power": str(np.round(np.random.normal(stats['Motor Apparent Power']['mean'], stats['Motor Apparent Power']['std']), 2)),
        "Motor Reactive Power": str(np.round(np.random.normal(stats['Motor Reactive Power']['mean'], stats['Motor Reactive Power']['std']), 2)),
        "Motor Shaft Power": str(np.round(np.random.normal(stats['Motor Shaft Power']['mean'], stats['Motor Shaft Power']['std']), 2)),
        "Motor Phase Current A": str(np.round(np.random.normal(stats['Motor Phase Current A']['mean'], stats['Motor Phase Current A']['std']), 2)),
        "Motor Phase Current B": str(np.round(np.random.normal(stats['Motor Phase Current B']['mean'], stats['Motor Phase Current B']['std']), 2)),
        "Motor Phase Current C": str(np.round(np.random.normal(stats['Motor Phase Current C']['mean'], stats['Motor Phase Current C']['std']), 2)),
        "Pump Thrust Bearing Active Temp": str(np.round(np.random.normal(stats['Pump Thrust Bearing Active Temp']['mean'], stats['Pump Thrust Bearing Active Temp']['std']), 2)),
        "Pump Inlet Pressure": str(np.round(np.random.normal(stats['Pump Inlet Pressure']['mean'], stats['Pump Inlet Pressure']['std']), 2)),
        "Pump Temp Unknown": str(np.round(np.random.normal(stats['Pump Temp Unknown']['mean'], stats['Pump Temp Unknown']['std']), 2)),
        "Pump Discharge Pressure 1": str(np.round(np.random.normal(stats['Pump Discharge Pressure 1']['mean'], stats['Pump Discharge Pressure 1']['std']), 2)),
        "Pump Discharge Pressure 2": str(np.round(np.random.normal(stats['Pump Discharge Pressure 2']['mean'], stats['Pump Discharge Pressure 2']['std']), 2))
    }
    return synthetic_data

if not data.empty:
    stats = calculate_stats(data)
else:
    stats = {}

def run_data_generation():
    while True:
        record_id = str(st.session_state.current_id)

        synthetic_data = generate_synthetic_data(record_id, stats)

        data_df = pd.DataFrame([synthetic_data])

        insert_data(data_df)

        st.write("Yeni Oluşturulan Veri:", synthetic_data)

        X_pred = data_df.drop(columns=['id', 'datetime']).astype(float).values.reshape(1, -1)
        prediction = model.predict(X_pred)

        binary_prediction = 1 if prediction[0][0] > 0.5 else 0
        st.write(f"Tahmin Sonucu (Binary): {binary_prediction}")

        save_predictions(record_id, binary_prediction)

        data = load_data()
        if not data.empty:
            data['datetime'] = pd.to_datetime(data['datetime'])
            fig = px.scatter(data, x='datetime', y='prediction', title='Biriken Veri Tahmin Grafiği')
            chart_placeholder.plotly_chart(fig)

        st.session_state.current_id += 1


        time.sleep(60)

run_data_generation()
