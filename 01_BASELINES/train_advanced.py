import pandas as pd
import numpy as np
import re
import os
import lightgbm as lgb
import pickle
from scipy.sparse import hstack, csr_matrix

# --- Load and Prepare Data ---
print("Loading data and ADVANCED features...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# --- Load ADVANCED Text Embeddings ---
with open('text_embeddings.pkl', 'rb') as f:
    text_embedding_dict = pickle.load(f)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
train_df = pd.merge(train_df, text_features_df, on='sample_id', how='left')
text_features_matrix = np.vstack(train_df['text_features'].values)

# --- Load Original Image and Dense Features ---
train_df['ipq'] = train_df['catalog_content'].apply(lambda x: 1) # Simplified
train_df['text_length'] = train_df['catalog_content'].apply(len)
with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
train_df['image_features'] = train_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix = np.vstack(train_df['image_features'].values)

# --- Combine ALL Features ---
y_train = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
# We combine dense features, NEW text features, and old image features
X_train_combined = np.hstack([dense_features.values, text_features_matrix, image_features_matrix])

# --- Train with Tuned Parameters ---
print("Training final model with advanced features...")
best_params = {
    'device': 'gpu',
    'objective': 'regression_l1',
    'metric': 'mae',
    'random_state': 42,
    'learning_rate': 0.023479730938523284,
    'num_leaves': 128,
    'max_depth': 41,
    'min_child_samples': 53,
    'subsample': 0.8372238305234515,
    'colsample_bytree': 0.793144061746289,
    'reg_alpha': 0.5393513849107224,
    'reg_lambda': 0.597106441202919
}
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train_combined, y_train)

# --- Save the new, more powerful model ---
print("Saving the new ADVANCED model...")
with open('advanced_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
    
print("\n✅ Advanced model saved successfully.")