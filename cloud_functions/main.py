import joblib
import numpy as np 
from flask import request
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.get_bucket("model-iris-data")
blob = bucket.blob("logistic_regression_v1.pkl")
blob.download_to_filename("/tmp/logistic_regression_v1.pkl")
model = joblib.load("/tmp/logistic_regression_v1.pkl")

def predict(request):
    data_json = request.get_json()
    
    sepal_length_cm = data_json["sepal_length_cm"]
    sepal_width_cm = data_json["sepal_width_cm"]
    petal_length_cm = data_json["petal_length_cm"]
    petal_width_cm = data_json["petal_width_cm"]

    data = np.array([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])
    predictions = model.predict(data)
    
    return str(predictions)