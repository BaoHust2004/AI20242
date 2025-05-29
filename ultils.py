# utils.py
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_processed_data():
    data = np.load('processed_data.npz')
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def evaluate_and_save(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    
    joblib.dump(model, f"models/{name}.pkl")
