import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import os
import xgboost as xgb
import catboost as cb

# --- 1. Define the SMAPE function ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

# --- 2. Load all trained models (base models + meta-model) ---
print("Loading all trained models...")
try:
    with open('lgbm_model.pkl', 'rb') as f: lgbm_model = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
    with open('cat_model.pkl', 'rb') as f: cat_model = pickle.load(f)
    with open('meta_model.pkl', 'rb') as f: meta_model = pickle.load(f)
except FileNotFoundError:
    print("Error: One or more model files not found. Please run 'train_ensemble.py' and 'train_stacking.py' first.")
    exit()

# --- 3. Load and prepare all data and features ---
print("Loading data and all feature sets...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# Load Features (Text Embeddings, Image Features, etc.)
with open('text_embeddings.pkl', 'rb') as f: text_embedding_dict = pickle.load(f)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
train_df = pd.merge(train_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(train_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)
train_df['ipq'] = 1
train_df['text_length'] = train_df['catalog_content'].fillna('').apply(len)
with open('image_features_FULL.pkl', 'rb') as f: image_features_dict = pickle.load(f)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
train_df['image_features'] = train_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(train_df['image_features'].values)

# Combine ALL Features
y = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_sparse = hstack([csr_matrix(dense_features.values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])
X_dense = X_sparse.toarray()

# --- 4. Recreate the exact same train/validation (holdout) split ---
X_train_s, X_val_s, y_train, y_val = train_test_split(X_sparse, y, test_size=0.2, random_state=42)
X_train_d, X_val_d, _, _ = train_test_split(X_dense, y, test_size=0.2, random_state=42)

# --- 5. Generate predictions from base models on the validation set ---
print("Making predictions on the validation set...")
lgbm_preds = lgbm_model.predict(X_val_s)
xgb_preds = xgb_model.predict(X_val_d)
cat_preds = cat_model.predict(X_val_s)

# --- 6. Create meta-features for the validation set ---
val_meta_features = np.vstack([lgbm_preds, xgb_preds, cat_preds]).T

# --- 7. Make final prediction with the new, smarter meta-model ---
final_preds_log = meta_model.predict(val_meta_features)

# Convert predictions and true values back to original price scale
preds_original = np.expm1(final_preds_log)
y_val_original = np.expm1(y_val.values)

# --- 8. Calculate and print the final SMAPE score ---
final_smape = smape(y_val_original, preds_original)

print("\n----------------------------------------------------------------")
print(f"Validation SMAPE for the ADVANCED STACKING Model: {final_smape:.4f}%")
print("----------------------------------------------------------------")