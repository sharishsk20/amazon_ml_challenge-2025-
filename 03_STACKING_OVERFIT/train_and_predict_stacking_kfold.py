import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- 1. Define the SMAPE function ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

print("Starting Advanced Stacking with K-Fold...")

# --- 2. Load and Prepare All Data and Features ---
print("Loading data and all feature sets...")
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

# Load Features (Text Embeddings, Image Features, etc.)
with open('text_embeddings.pkl', 'rb') as f: text_embedding_dict = pickle.load(f)
text_features_df = pd.DataFrame(text_embedding_dict.items(), columns=['sample_id', 'text_features'])
train_df = pd.merge(train_df, text_features_df, on='sample_id', how='left')
test_df = pd.merge(test_df, text_features_df, on='sample_id', how='left')
text_features_matrix_train = np.vstack(train_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)
text_features_matrix_test = np.vstack(test_df['text_features'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)).values)

train_df['ipq'] = 1
train_df['text_length'] = train_df['catalog_content'].fillna('').apply(len)
test_df['ipq'] = 1
test_df['text_length'] = test_df['catalog_content'].fillna('').apply(len)

with open('image_features_FULL.pkl', 'rb') as f: image_features_dict = pickle.load(f)
image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'image_features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
test_df = pd.merge(test_df, image_features_df, on='sample_id', how='left')
zero_feature_img = np.zeros(2048)
train_df['image_features'] = train_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix_train = np.vstack(train_df['image_features'].values)
test_df['image_features'] = test_df['image_features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature_img)
image_features_matrix_test = np.vstack(test_df['image_features'].values)

# Combine ALL Features
y = np.log1p(train_df['price'])
X_sparse = hstack([csr_matrix(train_df[['ipq', 'text_length']].values), csr_matrix(text_features_matrix_train), csr_matrix(image_features_matrix_train)])
X_test_sparse = hstack([csr_matrix(test_df[['ipq', 'text_length']].values), csr_matrix(text_features_matrix_test), csr_matrix(image_features_matrix_test)])
X_dense = X_sparse.toarray()
X_test_dense = X_test_sparse.toarray()


# --- 3. K-Fold Training Loop to Generate Meta-Features ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), 3)) # To store OOF predictions for meta-model
test_preds = np.zeros((len(test_df), 3)) # To store test predictions

lgbm_params = { 'device': 'gpu', 'objective': 'regression_l1', 'metric': 'mae', 'random_state': 42, 'learning_rate': 0.023479, 'num_leaves': 128, 'max_depth': 41, 'min_child_samples': 53, 'subsample': 0.83722, 'colsample_bytree': 0.79314, 'reg_alpha': 0.53935, 'reg_lambda': 0.59710 }
xgb_params = {'objective': 'reg:squarederror', 'device': 'cuda', 'random_state': 42}
cat_params = {'loss_function': 'MAE', 'task_type': 'GPU', 'random_seed': 42, 'verbose': 0}

for fold, (train_index, val_index) in enumerate(kf.split(X_sparse, y)):
    print(f"\n--- Processing Fold {fold+1}/5 ---")
    
    # Split data for this fold
    X_train_s, X_val_s = X_sparse[train_index], X_sparse[val_index]
    X_train_d, X_val_d = X_dense[train_index], X_dense[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train and predict with LightGBM
    lgbm = lgb.LGBMRegressor(**lgbm_params).fit(X_train_s, y_train)
    oof_preds[val_index, 0] = lgbm.predict(X_val_s)
    test_preds[:, 0] += lgbm.predict(X_test_sparse) / kf.n_splits
    
    # Train and predict with XGBoost
    xgb_m = xgb.XGBRegressor(**xgb_params).fit(X_train_d, y_train)
    oof_preds[val_index, 1] = xgb_m.predict(X_val_d)
    test_preds[:, 1] += xgb_m.predict(X_test_dense) / kf.n_splits
    
    # Train and predict with CatBoost
    cat = cb.CatBoostRegressor(**cat_params).fit(X_train_s, y_train)
    oof_preds[val_index, 2] = cat.predict(X_val_s)
    test_preds[:, 2] += cat.predict(X_test_sparse) / kf.n_splits

print("\nBase models trained and OOF predictions generated.")

# --- 4. Validate the Stacking Model using OOF Predictions ---
print("Training and validating the meta-model...")
meta_model = LinearRegression()
meta_model.fit(oof_preds, y)

oof_final_preds_log = meta_model.predict(oof_preds)
oof_final_preds = np.expm1(oof_final_preds_log)
y_original = np.expm1(y)
oof_smape = smape(y_original, oof_final_preds)

print("\n-------------------------------------------------------------")
print(f"OOF Validation SMAPE for STACKING Model: {oof_smape:.4f}%")
print("-------------------------------------------------------------")


## --- 5. Make Final Prediction on Test Set: K-Fold Averaging ---
print("Making final, robust K-FOLD AVERAGE prediction on the test set...")

# 'test_preds' already contains the average of the 5 folds for each of the 3 models
# Shape of test_preds is (num_samples, 3)

# Average the predictions across the three models
# (This is a robust average of 15 models in total: 5 folds * 3 models)
final_preds_log = test_preds.mean(axis=1) 
final_predictions = np.expm1(final_preds_log) # Inverse log transformg)

# --- 6. Save Submission ---
submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv('test_out_stacking_kfold.csv', index=False)

print("\n✅ Final STACKING K-FOLD submission file 'test_out_stacking_kfold.csv' created successfully!")