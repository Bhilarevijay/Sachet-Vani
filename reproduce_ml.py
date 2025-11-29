import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np
from predictor import predict_initial_case, refine_location_with_sightings, SCRIPT_DIR, MODEL_DIR_STG1, MODEL_DIR_STG2_PYTORCH

# Mock data for prediction
mock_input = {
    'abduction_time': 14.5,
    'latitude': 18.5204,
    'longitude': 73.8567,
    'age': 10,
    'gender': 'Male',
    'missing_date': '2023-10-27' 
}

print("--- Starting ML Verification ---")

print("\nTesting Initial Prediction...")
try:
    result = predict_initial_case(mock_input)
    print("Initial Prediction Result:")
    print(result)
except Exception as e:
    print(f"ERROR in Initial Prediction: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Refinement...")
try:
    initial_prediction = {
        'predicted_latitude': 18.5300,
        'predicted_longitude': 73.8600,
        'risk_label': 1
    }
    
    sightings = [
        {
            'lat': 18.5250,
            'lon': 73.8580,
            'hours_since': 1.0,
            'direction_text': 'Heading north'
        },
        {
            'lat': 18.5280,
            'lon': 73.8590,
            'hours_since': 2.0,
            'direction_text': 'Seen near the park'
        }
    ]
    
    lat, lon = refine_location_with_sightings(initial_prediction, sightings, mock_input)
    print(f"Refined Location: {lat}, {lon}")
except Exception as e:
    print(f"ERROR in Refinement: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Verification Complete ---")
