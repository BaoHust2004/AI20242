import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

# Tạo thư mục nếu chưa có
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load dữ liệu gốc
df1 = pd.read_csv('data/student-mat.csv', sep=';')
df2 = pd.read_csv('data/student-por.csv', sep=';')

# Merge 2 bảng
data = pd.concat([df1, df2], ignore_index=True)

# Scale G1, G2, G3 về [0–10] nếu đang là [0–20]
for col in ['G1', 'G2', 'G3']:
    data[col] = data[col] / 2.0

# Power transform cho G2
pt = PowerTransformer(method='yeo-johnson', standardize=False)
data['G2'] = pt.fit_transform(data[['G2']])
joblib.dump(pt, 'models/g2_power_transformer.pkl')

# Lưu file merge
data.to_csv('data/data_merge.csv', index=False)

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Chia input và target
X = data_encoded.drop('G3', axis=1)
y = data_encoded['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# Feature selection
selector_model = RandomForestRegressor(n_estimators=100, random_state=42)
selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
joblib.dump(selector, 'models/feature_selector.pkl')

# Lưu dữ liệu đã xử lý
np.savez('data/processed/processed_data.npz',
         X_train=X_train_selected, X_test=X_test_selected,
         y_train=y_train.to_numpy(), y_test=y_test.to_numpy())
