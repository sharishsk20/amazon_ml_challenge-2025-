import pandas as pd
import numpy as np
import re
import pickle
import os
from scipy.sparse import hstack, csr_matrix

# This script does not need GPU-specific libraries for prediction
print("Starting ENSEMBLE submission generation...")

# --- Load All Three Models ---
print("Loading ensemble models...")
try:
    with open('lgbm_model.pkl', 'rb') as f:
        lgbm_model = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('cat_model.pkl', 'rb') as f:
        cat_model = pickle.load(f)
except FileNotFoundError:
    print("Error: One or more model files not found. Please run 'train_ensemble.py' first.")
    exit()

# --- Load Pre-Generated Features ---
print("Loading pre-generated features...")
with open('text_embeddings.pkl', 'rb') as f:
    text_embedding_dict = pickle.load(f)
with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)

# --- Load and Process Test Data ---
DATASET_FOLDER = 'student_resource/dataset/'
test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

# --- Assemble Feature Matrix for Test Set ---
print("Assembling feature matrix for the test set...")

# Simple features
test_df['ipq'] = 1
test_df['text_length'] = test_df['catalog_content'].fillna('').apply(len)
dense_features = test_df[['ipq', 'text_length']]

# Text embedding features
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
test_df = pd.merge(test_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(test_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)

# Image features
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
test_df = pd.merge(test_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
test_df['image_features'] = test_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(test_df['image_features'].values)

# Combine everything into a dense array for compatibility with all models
X_test_combined = np.hstack([dense_features.values, text_features_matrix, image_features_matrix])

# --- Make Predictions with Each Model ---
print("Making predictions with each model...")
preds_lgbm = lgbm_model.predict(X_test_combined)
preds_xgb = xgb_model.predict(X_test_combined)
preds_cat = cat_model.predict(X_test_combined)

# --- Average the Predictions ---
print("Averaging the predictions...")
final_predictions_log = (preds_lgbm + preds_xgb + preds_cat) / 3.0
final_predictions = np.expm1(final_predictions_log)

# --- Create and Save the Submission File ---
submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv('test_out_ensemble.csv', index=False)

print("\n✅ Final ENSEMBLE submission file 'test_out_ensemble.csv' created successfully!")
