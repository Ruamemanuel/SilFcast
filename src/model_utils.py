import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def train_local_model(train_windows, test_window, thresholds, cv_folds, rf_params, svr_params, min_observations=6):
    predictions = []
    threshold_metrics = {threshold: {'count': 0, 'mse_sum': 0, 'errors': []} for threshold in thresholds}

    test_features = test_window[:-1].reshape(1, -1)
    test_target = test_window[-1]

    distances = cdist(test_features, train_windows.drop(columns='target').values, metric='cosine')
    similarities = 1 - distances[0]

    for threshold in thresholds:
        selected_windows = train_windows.iloc[similarities >= threshold]
        if len(selected_windows) >= min_observations:
            X_train = selected_windows.drop(columns='target')
            y_train = selected_windows['target']

            models = {
                'RF': RandomForestRegressor(random_state=42),
                'SVR': SVR(),
            }
            param_grids = {
                'RF': rf_params,
                'SVR': svr_params,
            }

            best_score = float('-inf')
            best_prediction = None

            for model_name, model in models.items():
                grid_search = GridSearchCV(model, param_grids[model_name], cv=cv_folds, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                current_score = -grid_search.best_score_

                if current_score > best_score:
                    best_score = current_score
                    best_prediction = grid_search.predict(test_features)[0]

            predictions.append(best_prediction)
            threshold_metrics[threshold]['count'] += 1
            threshold_metrics[threshold]['mse_sum'] += best_score
            threshold_metrics[threshold]['errors'].append(best_score)

    return predictions, threshold_metrics
