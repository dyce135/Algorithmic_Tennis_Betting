import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the new dataset
data = pd.read_csv("new_features.csv")

# Calculate the serve win percentage as the average of first serve win percentage and second serve win percentage
data['serve_win_percentage'] = (data['first_serve_win_percentage'] + data['second_serve_win_percentage']) / 2

# Drop 'first_serve_win_percentage' and 'second_serve_win_percentage' columns
data = data.drop(['first_serve_win_percentage', 'second_serve_win_percentage', 'player_id'], axis=1)

# Split the data into features and target variable
X = data.drop("serve_win_percentage", axis=1)
y = data["serve_win_percentage"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the GradientBoostingRegressor, RandomForestRegressor, and MLPRegressor models
xgb = GradientBoostingRegressor(random_state=42)
rfr = RandomForestRegressor(random_state=42)
mlp = MLPRegressor(random_state=42)

# Define the parameter grid for each model
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
rfr_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5]}
mlp_params = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'learning_rate_init': [0.001, 0.01, 0.1]}

# Perform Grid Search and Cross Validation to find the best parameters for each model
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
rfr_grid = GridSearchCV(rfr, rfr_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

xgb_grid.fit(X_train, y_train)
rfr_grid.fit(X_train, y_train)
mlp_grid.fit(X_train, y_train)

# Retrieve the best models
xgb_best = xgb_grid.best_estimator_
rfr_best = rfr_grid.best_estimator_
mlp_best = mlp_grid.best_estimator_

# Create the ensemble model with the best models
ensemble = VotingRegressor([('xgb', xgb_best), ('rfr', rfr_best), ('mlp', mlp_best)])

# Fit the ensemble model
ensemble.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble.predict(X_test)

# Calculate the performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test Mean Absolute Error: {mae}')
print(f'Test R^2 Score: {r2}')

joblib.dump(ensemble,'ensemble_model.pkl')
