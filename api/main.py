from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PredictionRequest, PredictionResponse, options, BatchPredictionRequest
from .predict import Predictor
import joblib

app = FastAPI(
    title="Sales Forecast API",
    description="Demo API for predicting sales using a simple ML model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Sales Forecast API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "options": "/options?type=country|category|device_type"
    }

@app.get("/options")
def get_options(type: str = None):
    '''Return available options for country, category, or device_type (or all if no type specified)'''
    if type:
        if type not in options:
            raise HTTPException(status_code=400, detail=f"Category must be one of {list(options.keys())}")
        return {type: options[type]}
    return options

@app.get("/health")
def health_check():
    '''Check if the model and options are loaded properly'''
    model, model_info = Predictor.load_model_details()
    model_check = model is not None
    options_check = options is not None
    model_info_check = model_info is not None
    if model_check and options_check and model_info_check:
        return {"status": "healthy"}
    else:
        return {
            "status": "unhealthy",
            "model_loaded": model_check,
            "options_loaded": options_check,
            "model_info_loaded": model_info_check
        }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict sales for a given date, product, and region.
    
    Returns predicted sales amount.
    """
    try:
        predictor = Predictor()
        prediction = predictor.predict_sales(
            date_str=request.date,
            country=request.country,
            category=request.category,
            device=request.device_type
        )
        
        return PredictionResponse(
            predicted_sales=prediction,
            date=request.date,
            country=request.country,
            category=request.category,
            device_type=request.device_type
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict/batch")
def batch_predict(request: BatchPredictionRequest):
    """
    Predict sales for a batch of requests.
    
    Returns list of predictions along with total and average forecasted value.
    """
    predictor = Predictor()
    predictions = []
    total_value = 0.0
    
    for req in request.requests:
        try:
            pred = predictor.predict_sales(
                date_str=req.date,
                country=req.country,
                category=req.category,
                device=req.device_type
            )
            predictions.append(PredictionResponse(
                predicted_sales=pred,
                date=req.date,
                country=req.country,
                category=req.category,
                device_type=req.device_type
            ))
            total_value += pred
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in request {req}: {str(e)}")
    
    average_value = total_value / len(request.requests) if request.requests else 0.0
    
    return {
        "predictions": predictions,
        "total_forecasted_value": round(total_value, 2),
        "average_forecasted_value": round(average_value, 2)
    }

@app.get("/model-info")
def get_model_info():
    """Get information about the model"""
    model, info = Predictor.load_model_details()
    return {
        "features": info['features'],
        "mappings": info['mappings'],
        "target": info['target'],
        "model_type": type(model).__name__,
        "info": "This model is a simple linear regression trained on synthetic data for demonstration purposes. It may not provide accurate predictions for real-world scenarios."
    }