# serve_model.py
import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn

app = FastAPI(title="IBM Telco Customer Churn Prediction Web Service")

MODEL_PATH = os.getenv("MODEL_PATH", "mlruns/0/latest/artifacts/model")

try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print(f"[+] Best model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[-] Error loading model: {e}")
    model = None


class CustomerInferenceInput(BaseModel):
    Gender: int
    Senior_Citizen: int = Field(..., alias="Senior Citizen")
    Partner: int
    Dependents: int
    Tenure_Months: float = Field(..., alias="Tenure Months")
    Phone_Service: int = Field(..., alias="Phone Service")
    Multiple_Lines_No_phone_service: int = Field(0, alias="Multiple Lines_No phone service")
    Multiple_Lines_Yes: int = Field(0, alias="Multiple Lines_Yes")
    Internet_Service_Fiber_optic: int = Field(0, alias="Internet Service_Fiber optic")
    Internet_Service_No: int = Field(0, alias="Internet Service_No")
    Online_Security_No_internet_service: int = Field(0, alias="Online Security_No internet service")
    Online_Security_Yes: int = Field(0, alias="Online Security_Yes")
    Online_Backup_No_internet_service: int = Field(0, alias="Online Backup_No internet service")
    Online_Backup_Yes: int = Field(0, alias="Online Backup_Yes")
    Device_Protection_No_internet_service: int = Field(0, alias="Device Protection_No internet service")
    Device_Protection_Yes: int = Field(0, alias="Device Protection_Yes")
    Tech_Support_No_internet_service: int = Field(0, alias="Tech Support_No internet service")
    Tech_Support_Yes: int = Field(0, alias="Tech Support_Yes")
    Streaming_TV_No_internet_service: int = Field(0, alias="Streaming TV_No internet service")
    Streaming_TV_Yes: int = Field(0, alias="Streaming TV_Yes")
    Streaming_Movies_No_internet_service: int = Field(0, alias="Streaming Movies_No internet service")
    Streaming_Movies_Yes: int = Field(0, alias="Streaming Movies_Yes")
    Contract_One_year: int = Field(0, alias="Contract_One year")
    Contract_Two_year: int = Field(0, alias="Contract_Two year")
    Paperless_Billing: int = Field(..., alias="Paperless Billing")
    Payment_Method_Credit_card_automatic: int = Field(0, alias="Payment Method_Credit card (automatic)")
    Payment_Method_Electronic_check: int = Field(0, alias="Payment Method_Electronic check")
    Payment_Method_Mailed_check: int = Field(0, alias="Payment Method_Mailed check")
    Monthly_Charges: float = Field(..., alias="Monthly Charges")
    Total_Charges: float = Field(..., alias="Total Charges")
    CLTV: float
    Number_of_Services: int
    Tenure_Group: int
    Monthly_to_Total_Ratio: float

    class Config:
        allow_population_by_field_name = True


@app.post("/predict")
def predict(data: CustomerInferenceInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Inference model is not loaded.")

    input_dict = data.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    prediction = int(model.predict(input_df)[0])
    try:
        probability = float(model.predict_proba(input_df)[0][1])
    except:
        probability = None

    return {
        "Churn_Prediction": prediction,
        "Churn_Probability": probability,
        "Status": "Churn" if prediction == 1 else "Active"
    }


@app.get("/")
def health_check():
    return {"status": "Active", "message": "Inference Web Service is active."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)