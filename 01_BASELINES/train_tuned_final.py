import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import os

print("Starting FINAL TUNED model training on GPU...")

# --- These are the optimal parameters you found ---
best_params = {
    'device': 'gpu',  # Make sure to use the GPU
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

# --- Load and Prepare 100% of the Training Data ---
DATASET_FOLDER = 'student_resource/dataset/'
train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
train_df['ipq'] = train_df['catalog_content'].apply(lambda x: int(re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I).group(1)) if re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I) else 1)
train_df['text_length'] = train_df['catalog_content'].apply(len)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_features = vectorizer.fit_transform(train_df['catalog_content'].fillna(''))

with open('image_features_FULL.pkl', 'rb') as f:
    image_features_dict = pickle.load(f)

image_features_df = pd.DataFrame(image_features_dict.items(), columns=['sample_id', 'features'])
train_df = pd.merge(train_df, image_features_df, on='sample_id', how='left')
zero_feature = np.zeros(2048)
train_df['features'] = train_df['features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature)
image_features_matrix = np.vstack(train_df['features'].values)

y_train = np.log1p(train_df['price'])
dense_features = train_df[['ipq', 'text_length']]
X_train_combined = hstack([csr_matrix(dense_features.values), tfidf_features, csr_matrix(image_features_matrix)])

# --- Train the Final Model on ALL data with the BEST parameters ---
print("Training final model on 100% of the data with tuned parameters...")
final_model = lgb.LGBMRegressor(**best_params)
# This will print an update every 10 trees the model builds
final_model.fit(X_train_combined, y_train, callbacks=[lgb.log_evaluation(period=10)])

# --- Save the Tuned Model and Vectorizer ---
print("Saving the tuned model and vectorizer...")
with open('tuned_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('tuned_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Tuned model saved successfully.")