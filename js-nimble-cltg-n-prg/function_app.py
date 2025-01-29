import azure.functions as func
import logging
import json
import requests
import pandas as pd
import os

app = func.FunctionApp()

@app.route(route="collecting_n_processing", auth_level=func.AuthLevel.ANONYMOUS)
def collecting_n_processing(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Fetching the Iris dataset...")

    try:
        iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

        df_iris = pd.read_csv(iris_url, names=column_names)

        csv_output_path = os.path.join(os.getcwd(), "iris_dataset.csv")
        df_iris.to_csv(csv_output_path, index=False)

        logging.info(f"Saved Iris dataset in {csv_output_path}")

        trigger_url = "http://localhost:7072/api/process_ml_pipeline"
        trigger_response = requests.post(trigger_url, json={"file_path": csv_output_path})

        return func.HttpResponse(
            json.dumps({"message": "Success", "csv_path": csv_output_path, "trigger_status": trigger_response.status_code}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal Server Error: {str(e)}", status_code=500)