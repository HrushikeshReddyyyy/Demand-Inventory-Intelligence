# Demand-Inventory-Intelligence


This project provides a demand forecasting solution for inventory management using historical sales data. The model uses time-series analysis, machine learning, and anomaly detection techniques to predict future demand and detect anomalies.

## Project Overview

This project uses historical sales data to forecast future sales using various forecasting models such as **Prophet** and **LSTM**. It includes:
- Data preprocessing and feature engineering
- Time-series forecasting using **Prophet** (for trend and seasonality modeling)
- Predictive modeling using **LSTM** (Long Short-Term Memory) networks
- Anomaly detection based on the forecasted sales units
- Optimal order quantity (EOQ) calculation based on forecasted data

## Libraries Used

This project utilizes several Python libraries for data manipulation, machine learning, and visualization:
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Prophet**: Time-series forecasting
- **TensorFlow**: Deep learning (LSTM model)
- **Shap**: Explainability of models using SHAP values
- **Scikit-learn**: Preprocessing and evaluation metrics
- **Streamlit**: Interactive web app for model visualization
- **Langchain**: For language models (optional, if using LLM-based analysis)

### Requirements

To run this project, you'll need to install the necessary libraries. You can install them using pip:

```bash
pip install pandas matplotlib prophet streamlit pyngrok langchain tensorflow shap scikit-learn bayesian-optimization transformers
Make sure you have a compatible Python version (preferably 3.x) and the necessary dependencies installed.

Dataset
The dataset used in this project is demand_forecasting.csv, which contains the following columns:

date: The date of the sale.

product_id: Identifier for the product.

category: Product category.

sales_units: Number of units sold.

sales_revenue: Total sales revenue.

Additional columns include promotion_applied, holiday_season, competitor_price_index, etc.

Make sure the dataset is in the same directory as the script or notebook to ensure the data is loaded properly.

Usage
Preprocessing: Clean the dataset and transform it for modeling. The dataset is loaded and preprocessed using Pandas.

Forecasting:

Prophet model: Provides predictions with confidence intervals. The model is trained on the historical sales data and predicts future sales.

LSTM model: Uses sequential data (sales units) to predict future values.

Anomaly Detection: Flags sales data that deviates significantly from the forecast using the Prophet model.

Optimal Order Quantity (EOQ): Calculates the optimal order quantity using the forecasted demand and given cost parameters.

You can run the project by executing the Demand-Inventory-Intelligence.ipynb notebook or using any Python IDE. Follow the instructions within the code cells.
