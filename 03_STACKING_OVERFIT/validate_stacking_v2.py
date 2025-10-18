import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import  os
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

print("Validating ADVANCED Stacking (v2) model...")

# --- Load All Models ---
with open('lgbm_model.pkl', 'rb') as f: lgbm_model = pickle.load(f)
with open('xgb_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
with open('cat_model.pkl', 'rb') as f: cat_model = pickle.load(f)
with open('meta_model_v2.pkl', 'rb') as f: meta_model_v2 = pickle.load(f)

# --- Load and Prepare Data ---
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
with open('text_embeddings.pkl', 'rb') as f: text_embedding_dict = pickle.load(f)
# ... (Same data loading as the training script)
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
y = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_sparse = hstack([csr_matrix(dense_features.values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])
X_dense = X_sparse.toarray()

# --- Recreate Splits ---
X_train_s, X_val_s, y_train, y_val = train_test_split(X_sparse, y, test_size=0.2, random_state=42)
X_train_d, X_val_d, _, _ = train_test_split(X_dense, y, test_size=0.2, random_state=42)
_, val_original_features, _, _ = train_test_split(dense_features, y, test_size=0.2, random_state=42)

# --- Generate Predictions ---
lgbm_preds = lgbm_model.predict(X_val_s)
xgb_preds = xgb_model.predict(X_val_d)
cat_preds = cat_model.predict(X_val_s)
base_preds = np.vstack([lgbm_preds, xgb_preds, cat_preds]).T

# --- Create Meta-Features with Context ---
val_meta_features = np.hstack([base_preds, val_original_features.values])

# --- Make Final Prediction ---
final_preds_log = meta_model_v2.predict(val_meta_features)
preds_original = np.expm1(final_preds_log)
y_val_original = np.expm1(y_val.values)

# --- Calculate and Print Scores ---
final_smape = smape(y_val_original, preds_original)
print("\n---------------------------------------------------------")
print(f"Validation SMAPE for Stacking v2 Model: {final_smape:.4f}%")
print("---------------------------------------------------------")