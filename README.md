# Demand Inventory Intelligence

This project provides a demand forecasting solution for inventory management using historical sales data. The model leverages time-series analysis, machine learning, and anomaly detection techniques to predict future demand and detect anomalies, ensuring optimal inventory levels and improved decision-making.

## 📝 Project Overview

This project utilizes historical sales data to forecast future sales using various advanced forecasting models such as **Prophet** and **LSTM**. The following key tasks are included in the project:

- **Data Preprocessing & Feature Engineering**: Cleaning and transforming the dataset for optimal model performance.
- **Time-series Forecasting with Prophet**: Predicting future sales based on trend and seasonality modeling.
- **Predictive Modeling using LSTM (Long Short-Term Memory)**: Using deep learning networks to predict future sales trends.
- **Anomaly Detection**: Identifying sales anomalies based on the forecasted sales units.
- **Optimal Order Quantity (EOQ) Calculation**: Determining the optimal order quantity to minimize inventory costs.

## ⚙️ Libraries Used

This project makes use of the following Python libraries for data manipulation, machine learning, and visualization:

- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For visualizing data.
- **Prophet**: For time-series forecasting (trend and seasonality modeling).
- **TensorFlow**: For building and training LSTM (Long Short-Term Memory) models.
- **SHAP**: For explainability of the models using SHAP values.
- **Scikit-learn**: For preprocessing and evaluation metrics.
- **Streamlit**: For building an interactive web app for model visualization.
- **Langchain**: For language models (optional, if using LLM-based analysis).

## 📊 Dataset

The dataset used in this project is `demand_forecasting.csv`, which includes the following columns:

- **`date`**: The date of the sale.
- **`product_id`**: The identifier for the product.
- **`category`**: The category of the product.
- **`sales_units`**: The number of units sold.
- **`sales_revenue`**: The total sales revenue.
- Additional columns include:
  - `promotion_applied`: Indicates if a promotion was applied.
  - `holiday_season`: Indicates whether it’s a holiday season.
  - `competitor_price_index`: Competitive pricing data for market comparison.

> **Note**: Ensure that the dataset is placed in the same directory as the script or notebook for proper data loading.

## 🛠️ Usage

Follow the steps below to run the project:

1. **Preprocessing**: Clean and transform the dataset using **Pandas** to ensure it's ready for modeling.
2. **Forecasting**:
   - **Prophet Model**: Uses historical data to predict future sales with confidence intervals.
   - **LSTM Model**: Utilizes sequential data (sales units) to forecast future sales.
3. **Anomaly Detection**: The **Prophet Model** flags sales data that deviates significantly from the forecast, helping to identify outliers.
4. **EOQ Calculation**: Computes the optimal order quantity based on forecasted demand and cost parameters to minimize costs.

### Example Usage:

```python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load data
df = pd.read_csv("demand_forecasting.csv")

# Preprocess and model
df_model = df.copy()
df_model['ds'] = pd.to_datetime(df_model['date'])
df_model['y'] = df_model['sales_units']

# Fit Prophet model
model = Prophet()
model.fit(df_model[['ds', 'y']])
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.show()
```

### 🚀 Results
1. After running the models, here’s what you can expect:
2. Prophet Model: Predicts future sales based on historical trends and seasonality, providing forecasts with confidence intervals.
3. LSTM Model: A deep learning model that forecasts future sales using sequential data (sales units).
4. Anomaly Detection: Flags significant deviations in sales, which can be reviewed for outliers or unusual events.
5. EOQ Calculation: Calculates the optimal order quantity based on the predicted demand, helping to minimize inventory costs.


