# Demand-Inventory-Intelligence


# Demand Inventory Intelligence

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
