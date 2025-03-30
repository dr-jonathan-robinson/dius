import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
import io
import uvicorn

# Define input models
class PredictionInput(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: str

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

# Initialize FastAPI app
app = FastAPI(
    title="DiUS ML Model API",
    description="API for making predictions with the DiUS TensorFlow model",
    version="1.0.0"
)

# Global variables
model = None
cat_mapping = None
numeric_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
categorical_cols = ['X10']

@app.get("/ping")
def ping():
    return {"status": "Yes master, whay is thy bidding?"}

@app.on_event("startup")
def load_model():
    """Load the trained model and categorical mapping on startup"""
    global model, cat_mapping
    
    model_path = os.environ.get('MODEL_PATH', 'dius_model.hdf5')
    mapping_path = os.environ.get('MAPPING_PATH', 'categorical_mapping.joblib')
    
    # Load the TensorFlow model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
    
    # Load the categorical mapping
    try:
        cat_mapping = joblib.load(mapping_path)
        print(f"Categorical mapping loaded successfully from {mapping_path}!")
    except Exception as e:
        print(f"Error loading categorical mapping: {str(e)}")
        cat_mapping = None

def prepare_input_data(input_df):
    """Prepare input data for model prediction"""
    # Handle categorical features
    input_df['X10_encoded'] = input_df['X10'].map(lambda x: cat_mapping.get(x, next(iter(cat_mapping.values()))))
    
    # Prepare inputs in the format expected by the model
    numeric_input = input_df[numeric_cols].values
    categorical_input = input_df['X10_encoded'].values.reshape(-1, 1)
    
    return {
        "numeric_inputs": numeric_input,
        "categorical_inputs": categorical_input
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is not None and cat_mapping is not None:
        return {"status": "healthy"}
    else:
        error_message = []
        if model is None:
            error_message.append("Model not loaded")
        if cat_mapping is None:
            error_message.append("Categorical mapping not loaded")
        
        raise HTTPException(
            status_code=500, 
            detail={"status": "unhealthy", "message": ", ".join(error_message)}
        )

@app.post("/predict", response_model=Dict[str, float])
def predict(data: PredictionInput):
    """Single prediction endpoint"""
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Prepare data for model
        model_input = prepare_input_data(input_df)
        
        # Make prediction
        prediction = model.predict(model_input).flatten()[0]
        
        # Return prediction
        return {"prediction": float(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=Dict[str, List[float]])
def batch_predict(data: BatchPredictionInput):
    """Batch prediction endpoint for JSON data"""
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([item.dict() for item in data.data])
        
        # Prepare data for model
        model_input = prepare_input_data(input_df)
        
        # Make prediction
        predictions = model.predict(model_input).flatten().tolist()
        
        # Return predictions
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict_csv")
async def batch_predict_csv(file: UploadFile = File(...)):
    """Batch prediction endpoint using CSV file"""
    try:
        # Read CSV file
        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))
        
        # Check if all required columns exist
        missing_cols = set(numeric_cols + categorical_cols) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns in input CSV: {', '.join(missing_cols)}"
            )
        
        # Prepare data for model
        model_input = prepare_input_data(input_df)
        
        # Make prediction
        predictions = model.predict(model_input).flatten().tolist()
        
        # Create results DataFrame
        results_df = pd.DataFrame({"prediction": predictions})
        
        # Convert to CSV string
        csv_output = io.StringIO()
        results_df.to_csv(csv_output, index=False)
        csv_output.seek(0)
        
        # Return CSV response
        return StreamingResponse(
            io.StringIO(csv_output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
