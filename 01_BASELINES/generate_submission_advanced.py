import pandas as pd
import numpy as np
import re
import pickle
import os
from tqdm import tqdm

# This script does not need GPU-specific libraries for prediction
print("Starting ADVANCED submission generation...")

# --- Load the ADVANCED Model ---
print("Loading the advanced pricing model...")
with open('advanced_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load the ADVANCED Text Embeddings ---
print("Loading pre-generated text embeddings...")
with open('text_embeddings.pkl', 'rb') as f:
    text_embedding_dict = pickle.load(f)

# --- Load the original Image and Dense Features ---
print("Loading pre-generated image features...")
with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)

# --- Load and Process Test Data ---
DATASET_FOLDER = 'student_resource/dataset/'
test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

# --- Assemble Feature Matrix for Test Set ---
print("Assembling final feature matrix for the test set...")
# Simple features
test_df['ipq'] = test_df['catalog_content'].apply(lambda x: int(re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I).group(1)) if re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I) else 1)
test_df['text_length'] = test_df['catalog_content'].apply(len)
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

# Combine everything
X_test_combined = np.hstack([dense_features.values, text_features_matrix, image_features_matrix])

# --- Make Final Predictions ---
print("Making final predictions with the advanced model...")
predictions_log = model.predict(X_test_combined)
final_predictions = np.expm1(predictions_log)

submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv('test_out_advanced.csv', index=False)

print("\n✅ Final ADVANCED submission file 'test_out_advanced.csv' created successfully!")
