import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle
import optuna

# --- 1. EVALUATION METRIC (SMAPE) ---
def smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# --- 2. DEFINE THE OBJECTIVE FUNCTION FOR OPTUNA ---
# This function now accepts the data as arguments
def objective(trial, X_train, y_train, X_val, y_val):
    """This function is called by Optuna to test a new set of hyperparameters."""
    
    # Define the search space for the hyperparameters
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'device': 'gpu',  # <-- THIS IS THE ONLY LINE THAT WAS ADDED
        'n_estimators': 2000, # Use a high number and rely on early stopping
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', -1, 64),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    # Train the model with the suggested parameters
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    # Make predictions and evaluate
    preds_log = model.predict(X_val)
    preds_original = np.expm1(preds_log)
    y_val_original = np.expm1(y_val)
    
    smape_score = smape(y_val_original, preds_original)
    
    return smape_score

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # --- DATA PREPARATION ---
    print("Preparing data for tuning...")
    
    # Load raw data
    DATASET_FOLDER = 'student_resource/dataset/' 
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Text features
    train_df['ipq'] = train_df['catalog_content'].apply(lambda x: int(re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I).group(1)) if re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I) else 1)
    train_df['text_length'] = train_df['catalog_content'].apply(len)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_features = vectorizer.fit_transform(train_df['catalog_content'].fillna(''))
    
    # Image features
    with open('image_features_FULL.pkl', 'rb') as f:
        image_features_dict = pickle.load(f)
    
    image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'features'])
    train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
    zero_feature = np.zeros(2048)
    train_df['features'] = train_df['features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature)
    image_features_matrix = np.vstack(train_df['features'].values)
    
    # Combine all features
    y = np.log1p(train_df['price'])
    dense_features = train_df[['ipq', 'text_length']]
    X_combined = hstack([csr_matrix(dense_features.values), tfidf_features, csr_matrix(image_features_matrix)])
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    print("Data preparation complete.")

    # --- OPTUNA STUDY ---
    print("Starting Optuna study (using GPU)...")
    
    study = optuna.create_study(direction='minimize')
    
    # Use a lambda function to pass the prepared data to the objective function
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)

    print("\nStudy finished!")
    print(f"Best SMAPE score: {study.best_value:.4f}%")
    print("Best hyperparameters found:")
    print(study.best_params)