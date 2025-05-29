# evaluate/evaluate.py
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
import importlib
import os
import pandas as pd
from visualize.visualize import plot_mae_comparison, plot_feature_importance

# Load dữ liệu đã xử lý
data = np.load("data/processed_data.npz")
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Danh sách mô hình và đường dẫn Python style
model_files = [
    ("Linear Regression", "training.train_linear"),
    ("Decision Tree", "training.train_decisiontree"),
    ("Random Forest", "training.train_randomforest"),
    ("XGBoost", "training.train_xgboost")
]

mae_scores = {}
best_model = None
best_mae = float('inf')
best_name = ""

# Đánh giá từng model
for name, module_path in model_files:
    module = importlib.import_module(module_path)
    model = module.train(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mae_scores[name] = mae

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

# Lưu model tốt nhất
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, f"model/best_model_{best_name.replace(' ', '_')}.pkl")

# Trực quan hóa
plot_mae_comparison(mae_scores)

# Trực quan hóa feature importance nếu hỗ trợ
if best_name in ['Random Forest', 'XGBoost']:
    df_encoded = pd.get_dummies(pd.read_csv("data/data_merge.csv", sep=','), drop_first=True)
    X_cols = df_encoded.drop("G3", axis=1).columns
    plot_feature_importance(best_model, X_cols.tolist(), best_name)
