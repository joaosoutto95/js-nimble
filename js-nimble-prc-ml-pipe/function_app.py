import azure.functions as func
import logging
import json
import requests
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


app = func.FunctionApp()

@app.route(route="process_ml_pipeline", auth_level=func.AuthLevel.ANONYMOUS)
def process_ml_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Starting the ML pipeline...")

    try:
        req_body = req.get_json()
        file_path = req_body.get("file_path")

        if not file_path or not os.path.exists(file_path):
            return func.HttpResponse("Invalid or missing file path.", status_code=400)

        df = pd.read_csv(file_path)

        le = LabelEncoder()
        df["species"] = le.fit_transform(df["species"])

        X = df.drop(columns=["species"])
        y = df["species"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        model_path = os.path.join(os.getcwd(), "iris_model.pkl")
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        metrics_path = os.path.join(os.getcwd(), "iris_model_metrics.json")
        with open(metrics_path, "w") as metrics_file:
            json.dump({"accuracy": accuracy, "classification_report": report}, metrics_file)

        predictions_path = os.path.join(os.getcwd(), "iris_predictions.csv")
        pd.DataFrame({"actual": y_test, "predicted": y_pred}).to_csv(predictions_path, index=False)

        feature_importance_path = os.path.join(os.getcwd(), "iris_feature_importance.csv")
        feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        feature_importance.to_csv(feature_importance_path, index=False)

        logging.info(f"Model training completed. Files saved:\n"
                     f"  - Model: {model_path}\n"
                     f"  - Metrics: {metrics_path}\n"
                     f"  - Predictions: {predictions_path}\n"
                     f"  - Feature Importance: {feature_importance_path}")

        return func.HttpResponse(
            json.dumps({
                "message": "Model trained successfully!",
                "accuracy": accuracy,
                "saved_files": {
                    "model": model_path,
                    "metrics": metrics_path,
                    "predictions": predictions_path,
                    "feature_importance": feature_importance_path
                }
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal Server Error: {str(e)}", status_code=500)
