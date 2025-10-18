import pandas as pd
import numpy as np
import re
import os
import pickle

print("Starting K-Fold ENSEMBLE submission generation...")

# --- Load all 5 K-Fold models ---
print("Loading all 5 trained models...")
models = []
for fold in range(1, 6):
    with open(f'model_fold_{fold}.pkl', 'rb') as f:
        models.append(pickle.load(f))
print(f"{len(models)} models loaded successfully.")

# --- Load Pre-Generated Features ---
print("Loading pre-generated features...")
with open('text_embeddings.pkl', 'rb') as f: text_embedding_dict = pickle.load(f)
with open('image_features_FULL.pkl', 'rb') as f: image_features_dict = pickle.load(f)

# --- Load and Process Test Data ---
DATASET_FOLDER = 'student_resource/dataset/'
test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

# --- Assemble Feature Matrix for Test Set ---
print("Assembling feature matrix...")
# ... (This part is the same as your previous submission script)
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
from scipy.sparse import hstack, csr_matrix
X_test = hstack([csr_matrix(test_df[['ipq', 'text_length']].values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])

# --- Make Predictions with Each Model and Average ---
print("Making predictions with each model...")
all_preds = []
for i, model in enumerate(models):
    print(f"Predicting with model {i+1}/5...")
    preds_log = model.predict(X_test)
    all_preds.append(preds_log)

# Average the predictions
print("Averaging the predictions...")
avg_preds_log = np.mean(all_preds, axis=0)
final_predictions = np.expm1(avg_preds_log)

# --- Create and Save the Submission File ---
submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv('test_out_kfold.csv', index=False)

print("\n✅ Final K-Fold submission file 'test_out_kfold.csv' created successfully!")
