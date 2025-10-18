import pandas as pd
import numpy as np
import re
import pickle
import os
from scipy.sparse import hstack, csr_matrix
from student_resource.src.utils import download_images
from tqdm import tqdm

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == '__main__':
    print("Verifying TensorFlow GPU access...")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected and available: {gpus}")
    else:
        print("❌ WARNING: TensorFlow did NOT detect a GPU. This script will be very slow.")

    print("\nStarting submission generation with TUNED model (BATCHED)...")

    from tensorflow.keras.preprocessing import image # type: ignore
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore

    print("Loading ResNet50 model for feature extraction...")
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    print("Loading the tuned pricing model and vectorizer...")
    with open('tuned_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tuned_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    DATASET_FOLDER = 'student_resource/dataset/'
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    print("\nDownloading test images...")
    TEST_IMAGE_FOLDER = 'test_images/'
    test_links = test_df['image_link'].dropna().unique()
    download_images(image_links=test_links, download_folder=TEST_IMAGE_FOLDER)

    print("\nExtracting features from test images using BATCHES...")
    test_image_features = {}
    BATCH_SIZE = 32 # Reduced for better memory management
    
    image_paths_to_process = []
    for index, row in test_df.iterrows():
        img_link = row['image_link']
        if isinstance(img_link, str):
            img_filename = os.path.basename(img_link)
            img_path = os.path.join(TEST_IMAGE_FOLDER, img_filename)
            if os.path.exists(img_path):
                image_paths_to_process.append({'sample_id': row['sample_id'], 'path': img_path})

    for batch_data in tqdm(batch_generator(image_paths_to_process, BATCH_SIZE)):
        batch_images = []
        batch_ids = []
        for item in batch_data:
            try:
                img = image.load_img(item['path'], target_size=(224, 224))
                img_array = image.img_to_array(img)
                batch_images.append(img_array)
                batch_ids.append(item['sample_id'])
            except Exception as e:
                print(f"\nWarning: Skipping corrupted image {item['path']}. Error: {e}")
                continue
        
        if batch_images:
            batch_images_np = np.array(batch_images)
            preprocessed_batch = preprocess_input(batch_images_np)
            batch_preds = resnet_model.predict(preprocessed_batch, verbose=0)
            
            for i, sample_id in enumerate(batch_ids):
                test_image_features[sample_id] = batch_preds[i]

    print("\nApplying final feature engineering...")
    test_df['ipq'] = test_df['catalog_content'].apply(lambda x: int(re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I).group(1)) if re.search(r'(?:pack of|set of|)(\d+)', str(x), re.I) else 1)
    test_df['text_length'] = test_df['catalog_content'].apply(len)
    tfidf_features = vectorizer.transform(test_df['catalog_content'].fillna(''))
    image_features_df = pd.DataFrame(test_image_features.items(), columns=['sample_id', 'features'])
    test_df = pd.merge(test_df, image_features_df, on='sample_id', how='left')
    zero_feature = np.zeros(2048)
    test_df['features'] = test_df['features'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_feature)
    image_features_matrix = np.vstack(test_df['features'].values)
    dense_features = test_df[['ipq', 'text_length']]
    X_test_combined = hstack([csr_matrix(dense_features.values), tfidf_features, csr_matrix(image_features_matrix)])

    print("\nMaking final predictions...")
    predictions_log = model.predict(X_test_combined)
    final_predictions = np.expm1(predictions_log)

    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
    submission_df.to_csv('test_out.csv', index=False)

    print("\n✅ Final submission file 'test_out.csv' created successfully!")