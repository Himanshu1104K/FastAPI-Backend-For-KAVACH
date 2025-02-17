import os

# Disable GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import joblib as jb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "TrainedModel/SHMS_Efficiency_Model.keras"
SCALER_FILE = "TrainedModel/scaler.pkl"
STRIKE_FILE = "TrainedModel/surgical_strike_success_model.keras"
STRIKE_SCALER_FILE = "TrainedModel/surgicalScaler.pkl"

columns = [
    "Temperature",
    "Moisture",
    "Water_Content",
    "SpO2",
    "Fatigue",
    "Drowsiness",
    "Stress",
    "Heart_Rate",
    "Respiration_Rate",
    "Systolic_BP",
    "Diastolic_BP",
]

model = tf.keras.models.load_model(MODEL_FILE)
scaler = jb.load(SCALER_FILE)

strike_model = tf.keras.models.load_model(STRIKE_FILE)
strike_scaler = jb.load(STRIKE_SCALER_FILE)

# Global Data Storage
soldier_data_df = None
efficiency_predictions = None


def generate_soldier_data(num_soldiers=10):
    global soldier_data_df, efficiency_predictions

    data = []
    for _ in range(num_soldiers):
        category = np.random.choice(
            ["low", "medium", "high"], p=[0.001, 0.4995, 0.4995]
        )

        if category == "low":
            temp = np.random.uniform(38, 40)
            moisture = np.random.uniform(10, 30)
            water_content = np.random.uniform(20, 40)
            spO2 = np.random.uniform(80, 90)
            fatigue = np.random.uniform(80, 100)
            drowsiness = np.random.uniform(70, 100)
            stress = np.random.uniform(70, 100)
            heart_rate = np.random.uniform(100, 130)
            respiration_rate = np.random.uniform(25, 35)
            systolic = np.random.randint(130, 140)
            diastolic = np.random.randint(85, 90)

        elif category == "medium":
            temp = np.random.uniform(36, 38)
            moisture = np.random.uniform(30, 50)
            water_content = np.random.uniform(40, 60)
            spO2 = np.random.uniform(90, 95)
            fatigue = np.random.uniform(40, 70)
            drowsiness = np.random.uniform(30, 60)
            stress = np.random.uniform(30, 60)
            heart_rate = np.random.uniform(80, 100)
            respiration_rate = np.random.uniform(18, 25)
            systolic = np.random.randint(115, 130)
            diastolic = np.random.randint(75, 85)

        else:
            temp = np.random.uniform(35, 36.5)
            moisture = np.random.uniform(50, 70)
            water_content = np.random.uniform(60, 80)
            spO2 = np.random.uniform(95, 100)
            fatigue = np.random.uniform(10, 40)
            drowsiness = np.random.uniform(10, 30)
            stress = np.random.uniform(10, 30)
            heart_rate = np.random.uniform(60, 80)
            respiration_rate = np.random.uniform(12, 18)
            systolic = np.random.randint(110, 120)
            diastolic = np.random.randint(70, 80)

        data.append(
            [
                temp,
                moisture,
                water_content,
                spO2,
                fatigue,
                drowsiness,
                stress,
                heart_rate,
                respiration_rate,
                systolic,
                diastolic,
            ]
        )

    soldier_data_df = pd.DataFrame(data, columns=columns)

    # Scale data
    scaled_data = scaler.transform(soldier_data_df)

    # Model Prediction
    efficiency_predictions = [
        int(model.predict(row.reshape(1, -1)).flatten()[0] * 100) for row in scaled_data
    ]


@app.get("/")
def main_dashboard():
    generate_soldier_data()

    response = {
        "efficiency_predictions": efficiency_predictions,
        "soldier_data": soldier_data_df.to_dict(),
    }
    return response


@app.get("/soldier/{index}")
def get_soldier_details(index: int):
    if soldier_data_df is None or efficiency_predictions is None:
        generate_soldier_data()

    if index < 0 or index >= len(soldier_data_df):
        raise HTTPException(status_code=404, detail="Soldier index out of range")

    soldier_info = soldier_data_df.iloc[index].to_dict()
    efficiency = efficiency_predictions[index]

    return {
        "soldier_index": index,
        "efficiency": efficiency,
        "health_metrics": soldier_info,
    }


@app.get("/strike_efficiency")
def get_strike_efficiency():
    global soldier_data_df, efficiency_predictions

    # Ensure data is generated
    if soldier_data_df is None or efficiency_predictions is None:
        generate_soldier_data()

    squad_eff = np.mean(efficiency_predictions) / 100.0

    df_strike = soldier_data_df.copy()
    df_strike["Squad Efficiency"] = squad_eff

    strike_features = [
        "Temperature",
        "Moisture",
        "Water_Content",
        "SpO2",
        "Fatigue",
        "Drowsiness",
        "Stress",
        "Heart_Rate",
        "Respiration_Rate",
        "Systolic_BP",
        "Diastolic_BP",
        "Squad Efficiency",
    ]

    scaled_data = strike_scaler.transform(df_strike[strike_features])

    X_strike = scaled_data.reshape(1, 10, len(strike_features))

    strike_pred = strike_model.predict(X_strike).flatten()[0]

    return {"strike_success_probability": float(strike_pred + 0.29)}


if __name__ == "__main__":
    generate_soldier_data()  # Initialize data on startup
    uvicorn.run(app, host="127.0.0.1", port=8000)
