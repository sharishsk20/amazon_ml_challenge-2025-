import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
from scipy.sparse import hstack, csr_matrix

print("Starting ADVANCED Stacking meta-model training...")

# --- Load Base Models ---
with open('lgbm_model.pkl', 'rb') as f: lgbm_model = pickle.load(f)
with open('xgb_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
with open('cat_model.pkl', 'rb') as f: cat_model = pickle.load(f)

# --- Load and Prepare Data ---
print("Loading data and features...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
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

y = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_sparse = hstack([csr_matrix(dense_features.values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])
X_dense = X_sparse.toarray()

# --- Create Holdout Set ---
X_train_s, X_holdout_s, y_train, y_holdout = train_test_split(X_sparse, y, test_size=0.2, random_state=42)
X_train_d, X_holdout_d, _, _ = train_test_split(X_dense, y, test_size=0.2, random_state=42)
# Keep the original dense features for the holdout set
_, holdout_original_features, _, _ = train_test_split(dense_features, y, test_size=0.2, random_state=42)

# --- Generate Base Predictions ---
print("Generating base model predictions...")
lgbm_oof_preds = lgbm_model.predict(X_holdout_s)
xgb_oof_preds = xgb_model.predict(X_holdout_d)
cat_oof_preds = cat_model.predict(X_holdout_s)
base_preds = np.vstack([lgbm_oof_preds, xgb_oof_preds, cat_oof_preds]).T

# --- NEW: Combine Base Predictions with Original Features ---
meta_features = np.hstack([base_preds, holdout_original_features.values])
print(f"Meta-features created with shape: {meta_features.shape}")

# --- Train the Smarter Meta-Model ---
print("Training the new meta-model...")
meta_model_v2 = lgb.LGBMRegressor(random_state=42)
meta_model_v2.fit(meta_features, y_holdout)

# Save the new manager model
with open('meta_model_v2.pkl', 'wb') as f: pickle.dump(meta_model_v2, f)
print("✅ Advanced stacking meta-model (v2) trained and saved.")