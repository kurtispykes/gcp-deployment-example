import joblib
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def run_training(): 
    """
    Train the model
    """
    # Read the training data 
    dataset = pd.read_csv(
        filepath_or_buffer="iris.data", 
        names=["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "class"]
        )

    # Split into labels and targets
    X = dataset.drop("class", axis=1).copy()
    y = dataset["class"].copy()

    # Create train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=26)

    # Training the model
    model = LogisticRegression(random_state=26)
    model.fit(X_train, y_train)

    # Persist the trained model 
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    run_training()

    