# End-to-End solution Nimble

## Setup Instructions (PowerShell code)

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/js-nimble.git
    ```
2. Navigate to the project directory:
    ```sh
    cd js-nimble
    ```
3. Install dependencies:
    ```sh
    
    ```
4. Run the application:
    ```sh
    
    ```

## Description of the Data Transformations

- **Data Collection**: Extracted for public API the airplane passanger stats throughout the years (1950 to 1960) and inserted in a dataframe.
- **Data Normalization**: Implemented a basic Min Max Scaler since this dataset has no outliers, but ideally in case of any outlier the code should do a Robust Scaling instead.
- **Feature Engineering**: Since it's a timeseries model I created 2 variables, the lag with 1 shift and the moving average with window size of 12, basically expecting a positive trend to be correlated to the target.

## Explanation of the Visualizations

The visualizations are generated using D3.js and can be found in the `visualizations` folder. The key visualizations include:
- **Bar Charts**: Representing categorical data.
- **Line Charts**: Showing trends over time.
- **Scatter Plots**: Displaying relationships between two variables.

## CI/CD Pipeline Description

The CI/CD pipeline is configured using GitHub Actions and is defined in the `.github/workflows` folder. The pipeline includes:
- **Build**: Compiling the project and running tests.
- **Test**: Executing unit and integration tests.
- **Deploy**: Automatically deploying the application to the production environment upon successful build and test stages.
