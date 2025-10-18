import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import pickle
import os
from sklearn.model_selection import train_test_split

# --- 1. Define the SMAPE function ---
def smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Add a small number to the denominator to avoid division by zero
    return np.mean(numerator / (denominator + 1e-8)) * 100

# --- 2. Load the final, advanced model ---
print("Loading the advanced model...")
try:
    with open('advanced_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'advanced_model.pkl' not found. Please run 'train_advanced.py' first.")
    exit()

# --- 3. Load all data and features (this mirrors the training script) ---
print("Loading data and all feature sets...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# Load advanced text embeddings
with open('text_embeddings.pkl', 'rb') as f:
    text_embedding_dict = pickle.load(f)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
train_df = pd.merge(train_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(train_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)

# Load original image and dense features
train_df['ipq'] = train_df['catalog_content'].apply(lambda x: int(re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I).group(1)) if re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I) else 1)
train_df['text_length'] = train_df['catalog_content'].apply(len)
with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
train_df['image_features'] = train_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(train_df['image_features'].values)

# --- 4. Combine all features into one matrix ---
y = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_combined = np.hstack([dense_features.values, text_features_matrix, image_features_matrix])

# --- 5. Recreate the EXACT same train/validation split as in tuning ---
# This ensures we are testing on data the model has not seen.
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- 6. Make predictions on the validation set ---
print("Making predictions on the validation set...")
preds_log = model.predict(X_val)

# Convert predictions and true values back to their original price scale
preds_original = np.expm1(preds_log)
y_val_original = np.expm1(y_val)

# --- 7. Calculate and print the final SMAPE score ---
final_smape = smape(y_val_original, preds_original)

print("\n------------------------------------------------------")
print(f"Validation SMAPE for the Advanced Model: {final_smape:.4f}%")
print("------------------------------------------------------")