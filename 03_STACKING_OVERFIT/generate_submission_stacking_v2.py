import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import hstack, csr_matrix

print("Starting ADVANCED STACKING (v2) submission generation...")

# --- 1. Load All Models (Base Models + v2 Meta-Model) ---
print("Loading all trained models...")
try:
    with open('lgbm_model.pkl', 'rb') as f: lgbm_model = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: xgb_model = pickle.load(f)
    with open('cat_model.pkl', 'rb') as f: cat_model = pickle.load(f)
    with open('meta_model_v2.pkl', 'rb') as f: meta_model_v2 = pickle.load(f)
except FileNotFoundError:
    print("Error: One or more model files not found. Please run 'train_ensemble.py' and 'train_stacking_v2.py' first.")
    exit()

# --- 2. Load and Prepare Test Data Features ---
print("Loading and preparing test data features...")
# (This section is the same as previous submission scripts)
with open('text_embeddings.pkl', 'rb') as f: text_embedding_dict = pickle.load(f)
with open('image_features_FULL.pkl', 'rb') as f: image_features_dict = pickle.load(f)
DATASET_FOLDER = 'student_resource/dataset/'
test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
test_df['ipq'] = 1
test_df['text_length'] = test_df['catalog_content'].fillna('').apply(len)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
test_df = pd.merge(test_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(test_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
test_df = pd.merge(test_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
test_df['image_features'] = test_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(test_df['image_features'].values)
dense_features = test_df[['ipq', 'text_length']]
X_test_sparse = hstack([csr_matrix(dense_features.values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])
X_test_dense = X_test_sparse.toarray()

# --- 3. Generate Predictions from Base Models on Test Set ---
print("Generating base predictions on the test set...")
lgbm_test_preds = lgbm_model.predict(X_test_sparse)
xgb_test_preds = xgb_model.predict(X_test_dense)
cat_test_preds = cat_model.predict(X_test_sparse)
base_preds = np.vstack([lgbm_test_preds, xgb_test_preds, cat_test_preds]).T

# --- 4. Create Meta-Features for the Test Set (with original features) ---
test_meta_features = np.hstack([base_preds, dense_features.values])

# --- 5. Make Final Prediction with the v2 Meta-Model ---
print("Making final prediction with the 'smarter manager' model...")
final_preds_log = meta_model_v2.predict(test_meta_features)
final_predictions = np.expm1(final_preds_log)

# --- 6. Save the Submission File ---
submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv('test_out_stacking_v2.csv', index=False)

print("\n✅ Final ADVANCED STACKING (v2) submission file 'test_out_stacking_v2.csv' created successfully!")