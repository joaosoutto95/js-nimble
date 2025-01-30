# End-to-End solution Nimble

## Setup Instructions (Running locally):

1. Clone the repository;
2. Download and Setup Azure Functions Core Tools;
3. Navigate to the folder solution-part2 inside the project root folder;
4. Create and Activate a virtual enviroment;
5. Pip install the requirements.txt;
6. Use Azure Function command to run router;
    ```sh
    func start
    ```
7. Check if function is running correctly;
8. Make a request to the endpoint http://localhost:7071/api/process-data
    ```sh
    Invoke-WebRequest -Uri http://localhost:7071/api/process-data -Method Post
    ```
9. After finishing running, the function will save 5 files starting with output_*, this are the results of the API request and model training;
10. In the folder solution-part3 there is a .pbix file that ingest the data from the output files and plots analysis on dataset and model training results.

## Description of the Data Transformations

- **Data Collection**: Extracted for public API the airplane passanger stats throughout the years (1950 to 1960) and inserted in a dataframe.
- **Data Normalization**: Implemented a basic Min Max Scaler since this dataset has no outliers, but ideally in case of any outlier the code should do a Robust Scaling instead.
- **Feature Engineering**: Since it's a timeseries model I created 2 variables, the lag with 1 shift and the moving average with window size of 12, basically expecting a positive trend to be correlated to the target.

## Explanation of the Visualizations

The .pbx file inside solution-part3 folder has 2 pages:
    - **FIRST PAGE**: 
        - Plot with the distribution of the whole Air Passengers dataset that was used for training;
        - Some KPI's as well that could help monitor the data quality that is being ingested on the model.
    - **SECOND PAGE**: 
        - Plot showing the Predicted values vs. Actual to show how well fitted the new trained model is;
        - Model performance metrics on cards - MAPE, MAE, MSE, and RMSE;
        - Feature Importance plot to show the corrilation on training.

## CI/CD Pipeline Description

The CI/CD pipeline is configured using Azure Pipelines. The pipeline includes:
- Code Linting: Runs flake8 to enforce Python style guidelines;
- Unit Tests & Coverage: Executes pytest with --cov to ensure test coverage.
