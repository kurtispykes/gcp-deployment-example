import joblib 
import pandas as pd

model = joblib.load("logistic_regression_v1.pkl")

def make_prediction(inputs): 
    """
    Make a prediction using the trained model 
    """
    inputs_df = pd.DataFrame(
        inputs, 
        columns=["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
        )
    predictions = model.predict(inputs_df)
    
    return predictions