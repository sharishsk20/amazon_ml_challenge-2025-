import pandas as pd
import numpy as np
import re
import pickle
import os
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

print("Starting ENSEMBLE model training...")

# --- Load and Prepare 100% of the Training Data ---
print("Loading data and all feature sets...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# Load Advanced Text Embeddings
with open('text_embeddings.pkl', 'rb') as f:
    text_embedding_dict = pickle.load(f)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
train_df = pd.merge(train_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(train_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)

# Load Image and Dense Features
train_df['ipq'] = train_df['catalog_content'].apply(lambda x: 1)
train_df['text_length'] = train_df['catalog_content'].apply(len)
with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
train_df['image_features'] = train_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(train_df['image_features'].values)

# Combine ALL Features
y_train = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_train_combined = np.hstack([dense_features.values, text_features_matrix, image_features_matrix])

# --- Train Model 1: LightGBM (Your Tuned Champion) ---
print("\nTraining LightGBM model...")
lgbm_params = {
    'device': 'gpu', 'objective': 'regression_l1', 'metric': 'mae', 'random_state': 42,
    'learning_rate': 0.023479, 'num_leaves': 128, 'max_depth': 41, 'min_child_samples': 53,
    'subsample': 0.83722, 'colsample_bytree': 0.79314, 'reg_alpha': 0.53935, 'reg_lambda': 0.59710
}
lgbm_model = lgb.LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train_combined, y_train)
with open('lgbm_model.pkl', 'wb') as f: pickle.dump(lgbm_model, f)
print("LightGBM model saved.")

# --- Train Model 2: XGBoost ---
print("\nTraining XGBoost model...")
xgb_params = {'objective': 'reg:squarederror', 'tree_method': 'gpu_hist', 'random_state': 42}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train_combined, y_train)
with open('xgb_model.pkl', 'wb') as f: pickle.dump(xgb_model, f)
print("XGBoost model saved.")

# --- Train Model 3: CatBoost ---
print("\nTraining CatBoost model...")
cat_params = {'loss_function': 'MAE', 'task_type': 'GPU', 'random_seed': 42, 'verbose': 0}
cat_model = cb.CatBoostRegressor(**cat_params)
cat_model.fit(X_train_combined, y_train)
with open('cat_model.pkl', 'wb') as f: pickle.dump(cat_model, f)
print("CatBoost model saved.")

print("\n✅ All ensemble models trained and saved successfully.")