import joblib
import json
import pandas as pd
from datetime import datetime
import numpy as np


    
class Predictor():
    def __init__(self, model=None):
        if model is not None:
            self.model = model
            _, self.model_info = self.load_model_details()
            if self.model_info is None:
                raise Exception("Model info could not be loaded")
        else:
            self.model, self.model_info = self.load_model_details()
            if self.model is None or self.model_info is None:
                raise Exception("Model or model info could not be loaded")
        
    def load_model_details(self):
        try:
            model = joblib.load('model/sales_model.pkl')
            model_info = joblib.load('model/model_info.pkl')
            return model, model_info
        except Exception as e:
            print(f"Error loading model or info: {e}")
            return None, None

    def predict_sales(self, date_str: str, country: str, category: str, device: str) -> float:
        """Make a sales prediction (returns predicted sales amount)"""
        
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        day_of_year = date_obj.timetuple().tm_yday
        day_of_week = date_obj.weekday()
        month = date_obj.month
        year = date_obj.year
        
        country_mapping = {v: k for k, v in self.model_info['mappings']['country'].items()}
        category_mapping = {v: k for k, v in self.model_info['mappings']['category'].items()}
        device_mapping = {v: k for k, v in self.model_info['mappings']['device_type'].items()}

        try:
            country_encoded = country_mapping.get(country)
            category_encoded = category_mapping.get(category)
            device_encoded = device_mapping.get(device)
            if country_encoded is None:
                raise ValueError(f"Country '{country}' not recognized. Valid options: {list(country_mapping.keys())}")
            if category_encoded is None:
                raise ValueError(f"Category '{category}' not recognized. Valid options: {list(category_mapping.keys())}")
            if device_encoded is None:
                raise ValueError(f"Device type '{device}' not recognized. Valid options: {list(device_mapping.keys())}")
        except Exception as e:
            raise ValueError(str(e))
        
        features = [[day_of_year, month, year, country_encoded, category_encoded, device_encoded, day_of_week]]
        
        # Predict
        prediction = np.expm1(self.model.predict(features)[0])
        
        return round(prediction, 2)