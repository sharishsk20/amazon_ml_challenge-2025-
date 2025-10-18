import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os
import numpy as np

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("Loading pre-trained sentence-transformer model...")
    # This is a powerful, lightweight model perfect for this task.
    # It will be downloaded automatically on the first run.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the training data
    DATASET_FOLDER = 'student_resource/dataset/'
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

    # We need to process the text from both train and test sets
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    all_text = pd.concat([train_df['catalog_content'], test_df['catalog_content']]).fillna(' ').tolist()
    all_ids = pd.concat([train_df['sample_id'], test_df['sample_id']]).tolist()

    print(f"Generating embeddings for {len(all_text)} text samples...")

    # Generate embeddings in batches for efficiency. The model handles GPU usage automatically.
    text_embeddings = model.encode(all_text, show_progress_bar=True, batch_size=128)

    # Save the embeddings to a file
    embedding_dict = {sample_id: embedding for sample_id, embedding in zip(all_ids, text_embeddings)}

    output_filename = 'text_embeddings.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(embedding_dict, f)

    print(f"\n✅ Text embeddings saved successfully to '{output_filename}'.")