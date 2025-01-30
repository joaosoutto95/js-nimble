import azure.functions as func
import pandas as pd
import logging
import json
import requests
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, root_mean_squared_error


app = func.FunctionApp()

@app.route(route="process-data", auth_level=func.AuthLevel.ANONYMOUS)
def process_data(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Getting the requested data...")

    try:
        # Got a common example dataset from the internet
        dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        
        data = pd.read_csv(dataset_url)
        data['month'] = pd.to_datetime(data['Month'])
        data['passengers'] = data['Passengers']
        data[['month', 'passengers']].set_index('month', inplace=True)

        logging.info("Dataset extracted and transformed.")
        
        csv_output_path = os.path.join(os.getcwd(), "output_file_airline_passengers.csv")
        data[['month', 'passengers']].to_csv(csv_output_path, index=False)

        logging.info(f"Data saved to {csv_output_path}")

        trigger_url = "http://localhost:7071/api/process-ml-pipeline"
        trigger_response = requests.post(trigger_url, json={"file_path": csv_output_path})

        return func.HttpResponse(
            json.dumps({"message": "Success", "trigger_status": trigger_response.status_code}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal Server Error: {str(e)}", status_code=500)


@app.route(route="process-ml-pipeline", auth_level=func.AuthLevel.ANONYMOUS)
def process_ml_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Starting ML pipeline...")

    try:
        req_body = req.get_json()
        file_path = req_body.get("file_path")

        if not file_path or not os.path.exists(file_path):
            return func.HttpResponse("File path is invalid or missing.", status_code=400)

        data = pd.read_csv(file_path, parse_dates=['month'], index_col='month')

        data['t-1'] = data['passengers'].shift(1)
        data['rolling_mean'] = data['passengers'].rolling(window=12).mean()

        data.dropna(inplace=True)

        test_size = 12
        train_data = data[:-test_size]
        test_data = data[-test_size:]

        X_train = train_data[['t-1', 'rolling_mean']]
        y_train = train_data['passengers']

        X_test = test_data[['t-1', 'rolling_mean']]
        y_test = test_data['passengers']

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_int = y_pred.astype(int)

        mape = mean_absolute_percentage_error(y_test, y_pred_int)
        mae = mean_absolute_error(y_test, y_pred_int)
        mse = mean_squared_error(y_test, y_pred_int)
        rmse = root_mean_squared_error(y_test, y_pred_int)

        logging.info(f"Model succesfully trained. MAPE on test data: {mape} %")

        model_file_path = os.path.join(os.getcwd(), "output_file_regression_model.pkl")
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)

        metrics = {
            'mape': mape,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
        }

        metrics_file_path = os.path.join(os.getcwd(), "output_file_model_metrics.json")
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f)

        predictions_df = pd.DataFrame({'month':y_test.index, 'actual': y_test, 'predicted': y_pred_int})
        predictions_file_path = os.path.join(os.getcwd(), "output_file_model_predictions.csv")
        predictions_df.to_csv(predictions_file_path, index=False)

        logging.info(f"Model and metrics saved.")
        logging.info(f"Predictions saved to {predictions_file_path}")

        feature_importance = model.coef_
        feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns, columns=['importance']).reset_index()
        feature_importance_df = feature_importance_df.rename(columns={'index': 'feature'})
        feature_importance_file = os.path.join(os.getcwd(), "output_file_feature_importance.csv")
        feature_importance_df.to_csv(feature_importance_file, index=False)
    
        logging.info(f"Model training completed. Files saved:\n"
                     f"  - Model: {model_file_path}\n"
                     f"  - Metrics: {metrics_file_path}\n"
                     f"  - Predictions: {predictions_file_path}\n"
                     f"  - Feature Importance: {feature_importance_file}")

        return func.HttpResponse(
            json.dumps({
                "message": "Model trained successfully!",
                "saved_files": {
                    "model": model_file_path,
                    "metrics": metrics_file_path,
                    "predictions": predictions_file_path,
                    "feature_importance": feature_importance_file
                }
            }),
            mimetype="application/json",
            status_code=200
        )    
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal Server Error: {str(e)}", status_code=500)