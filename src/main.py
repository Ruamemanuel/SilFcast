import pandas as pd
from src.data_utils import get_sliding_windows, normalize_series, cosine_similarity
from src.model_utils import train_local_model
from src.evaluation import smape, rmse
from sklearn.model_selection import train_test_split

# Load and normalize data
data = pd.read_csv('data/your_dataset.csv')
series_name = 'example_series'
data_series = data[series_name]
train_size = 0.75
window_size = 12
thresholds = [0.75, 0.8, 0.85, 0.9]

data_norm, scaler = normalize_series(data_series)
train_norm, test_norm = train_test_split(dados_norm, train_size=train_size, shuffle=False, random_state=42)

train_windows = get_sliding_windows(window_size, train_series)
test_windows = get_sliding_windows(window_size, pd.concat([train_series, test_series]))

rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10, 15, 20]}
svr_params = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10], 'epsilon': [0.1, 0.01, 0.001], 'gamma': [1, 0.1, 0.01, 0.001]}

# Local training and evaluation
predictions = []
for test_window in test_windows.iloc[-len(test_series):, :].values:
    preds, metrics = train_local_model(train_windows, test_window, thresholds, cv_folds=3, rf_params=rf_params, svr_params=svr_params)
    predictions.append(preds)

# Calculate metrics
print('SMAPE:', smape(test_series, predictions))
print('RMSE:', rmse(test_series, predictions))
