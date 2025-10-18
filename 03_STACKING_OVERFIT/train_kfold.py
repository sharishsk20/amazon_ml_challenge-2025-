import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.model_selection import KFold
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix

# --- 1. Define the SMAPE function ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

print("Starting K-Fold model training with OOF validation...")

# --- Your best hyperparameters ---
best_params = {
    'device': 'gpu', 'objective': 'regression_l1', 'metric': 'mae', 'random_state': 42,
    'learning_rate': 0.023479, 'num_leaves': 128, 'max_depth': 41, 'min_child_samples': 53,
    'subsample': 0.83722, 'colsample_bytree': 0.79314, 'reg_alpha': 0.53935, 'reg_lambda': 0.59710
}

# --- Load and Prepare Data ---
print("Loading data and all feature sets...")
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
X = hstack([csr_matrix(dense_features.values), csr_matrix(text_features_matrix), csr_matrix(image_features_matrix)])

# --- K-Fold Training Loop with Score Tracking ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_smape_scores = [] # NEW: List to store scores from each fold

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n--- Training Fold {fold+1}/5 ---")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # --- NEW: Calculate and store the validation score for this fold ---
    preds_log = model.predict(X_val)
    preds_original = np.expm1(preds_log)
    y_val_original = np.expm1(y_val)
    
    fold_smape = smape(y_val_original, preds_original)
    oof_smape_scores.append(fold_smape)
    print(f"Fold {fold+1} SMAPE: {fold_smape:.4f}%")
    # --- END NEW PART ---
    
    with open(f'model_fold_{fold+1}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Fold {fold+1} model saved.")

# --- NEW: Print the final average score ---
print("\n------------------------------------------------------")
print(f"Average K-Fold Validation SMAPE: {np.mean(oof_smape_scores):.4f}%")
print("------------------------------------------------------")