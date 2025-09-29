# Sales Forecasting with Random Forests

This repository provides an end-to-end pipeline for daily sales revenue forecasting using a Random Forest Regressor, leveraging merged order and product datasets. The workflow includes feature engineering, model tuning, evaluation, and export for deployment[web:1].

---

### Features

- Aggregates daily sales revenue for each product[web:1].
- Includes calendar, holiday, lagged, and rolling features to enhance prediction[web:1].
- Performs one-hot encoding for categorical product attributes[web:1].
- Applies standard scaling before modeling[web:1].
- Uses Random Forests with hyperparameter optimization via randomized search[web:1].
- Evaluates model performance (MAE, MSE, R²) and visualizes actual vs. predicted revenue[web:1].
- Displays feature importance for interpretability[web:1].
- Serializes trained model and scaler for future use[web:1].

---

### Requirements

- Python >=3.7
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle

Install dependencies: pip install pandas numpy scikit-learn matplotlib


---

### Data

The script expects a CSV named `orders_products_merged.csv`, including these columns:

- `order_date` (date)
- `product_id` (identifier)
- `title` (string)
- `price` (float)
- `product_type` (category)
- `vendor` (category)
- `total` (daily sales revenue for aggregation)

---

### Usage

1. Place `orders_products_merged.csv` in your working directory.
2. Run the script:
    ```
    python sales_forecast.py
    ```
3. The script will:
    - Engineer features (lags, rolling windows, holidays, categorical encoding)[web:1].
    - Split chronologically into train/test sets[web:1].
    - Tune hyperparameters with time-series cross-validation[web:1].
    - Output key metrics and actual vs. predicted plots[web:1].
    - Save the best model and the scaler:
        - `sales_forecast_model.pkl`
        - `feature_scaler.pkl`

---

### Model Details

- **Model:** RandomForestRegressor (scikit-learn)[web:1]
- **CV:** TimeSeriesSplit (preserves temporal order)[web:1]
- **Hyperparameters Tuned:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features[web:1]
- **Feature Engineering:**
    - Lags (1, 3, 7, 14, 28 days)[web:1]
    - Rolling mean and std (3, 7, 14, 28 days)[web:1]
    - Day of week, month, week of year[web:1]
    - Holiday indicator (New Year, Valentine's, Thanksgiving, Christmas)[web:1]
    - Per-product averages and standard deviations[web:1]
    - One-hot encoding for product type and vendor[web:1]

---

### Evaluation

- **Metrics:**
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - R² Score
- **Visualization:**
    - Actual vs Predicted daily sales
- **Feature importance:**
    - Top predictors displayed in output

---

### Deployment

After training, the serialized model and scaler can be loaded for real-time or batch forecasting:

import pickle

with open('sales_forecast_model.pkl', 'rb') as f:
model = pickle.load(f)

with open('feature_scaler.pkl', 'rb') as f:
scaler = pickle.load(f)

Predict on new scaled data
y_pred = model.predict(scaler.transform(new_X))


---

### License

MIT License[web:1].
