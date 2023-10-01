# UTT AMIS NET ASSET VALUE FORECASTING

This project deals with predicting the Net Asset Value for a certain Portifolio company which
will the investor to invest more in their funds

## INSTALLATION

**To use the code and run the forecasting models, follow these installation steps:**

1. **Clone the repository to your local machine:**

```
git clone https://github.com/Godie360/Net-Asset-Value-Forecasting.git
```

```
cd utt-amis-net-asset-forecasting
```

```
pip install -r requirements.txt
```

## DATASET

### Data Sources

The data sources used in the project and provide details are from UTT AMIS Performance Funds

-   **Data Source** :UTT AMIS Performance Funds
    -   Location: [URL]
    -   Format: CSV
    -   Permissions: Access granted under [(Tanzania Data Lab)]

### Data Preprocessing

-   Data cleaning to remove outliers and errors.
-   Feature engineering to create relevant variables.

## METHODOLOGY

Modethodology used is CRISP (Cross Industry Standard Process )

### FORECASTING METHODOLOGY

-   We used a time series forecasting approach, specifically the **Facebook Prophet** model, to predict net assets value.
-   The Facebook Prophet model was trained on historical net asset data and tuned using grid search for optimal hyperparameters.
-   Additionally, we experimented with a machine learning approach, such as a XGBoost, to compare performance.
