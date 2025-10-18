import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == '__main__':
    print("Verifying TensorFlow GPU access...")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ WARNING: TensorFlow did NOT detect a GPU.")
    else:
        print(f"✅ GPU Detected: {gpus}")

    print("Loading pre-trained EfficientNetB3 model...")
    main_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    
    IMAGE_FOLDER = 'images/'
    TEST_IMAGE_FOLDER = 'test_images/'
    DATASET_FOLDER = 'student_resource/dataset/'
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    all_image_data = pd.concat([
        train_df[['sample_id', 'image_link']],
        test_df[['sample_id', 'image_link']]
    ]).dropna().drop_duplicates()

    image_paths_to_process = []
    for index, row in all_image_data.iterrows():
        img_filename = os.path.basename(row['image_link'])
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(TEST_IMAGE_FOLDER, img_filename)
        if os.path.exists(img_path):
            image_paths_to_process.append({'sample_id': row['sample_id'], 'path': img_path})
            
    print(f"\nExtracting features from {len(image_paths_to_process)} images using BATCHES...")
    image_features = {}
    BATCH_SIZE = 32 # Reduced for better memory management

    for batch_data in tqdm(batch_generator(image_paths_to_process, BATCH_SIZE)):
        batch_images = []
        batch_ids = []
        for item in batch_data:
            try:
                img = image.load_img(item['path'], target_size=(300, 300))
                img_array = image.img_to_array(img)
                batch_images.append(img_array)
                batch_ids.append(item['sample_id'])
            except Exception:
                continue
        
        if batch_images:
            batch_images_np = np.array(batch_images)
            preprocessed_batch = preprocess_input(batch_images_np)
            batch_preds = main_model.predict(preprocessed_batch, verbose=0)
            
            for i, sample_id in enumerate(batch_ids):
                image_features[sample_id] = batch_preds[i]

    output_filename = 'image_features_EFFNET.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(image_features, f)

    print(f"\n✅ EfficientNet feature extraction complete. Features saved to '{output_filename}'.")