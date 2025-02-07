import tensorflow as tf
import joblib as jb
import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Frontend domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

MODEL_FILE = "TrainedModel/SHMS_Efficiency_Model.keras"
SCALER_FILE = "TrainedModel/scaler.pkl"

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


@app.get("/")
def generate_data():
    data = []
    for _ in range(10):
        category = np.random.choice(["low", "medium", "high"], p=[0.001, 0.4, 0.49])

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

    df = pd.DataFrame(data, columns=columns)

    # Scale data (remove timestamp before scaling)
    scaled_data = scaler.transform(df)

    # Model Prediction
    predictions = []
    for row in scaled_data:
        input_data = row.reshape(1, -1)
        predictions.append(model.predict(input_data).flatten()[0])

    # Convert NumPy array to list for JSON response
    predictions = [int(prediction * 100) for prediction in predictions]
    response = {"efficiency_predictions": predictions}

    # 10 Soldier Data
    soldier_data = df.to_dict()
    response["soldier_data"] = soldier_data

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
