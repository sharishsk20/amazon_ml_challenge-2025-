import os
import signal
import sys
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm
import urllib.request

# --- NEW: Define a global pool variable ---
# This allows our signal handler to access and terminate the pool.
pool = None

# --- NEW: Define the emergency stop function ---
def signal_handler(sig, frame):
    """This function runs when Ctrl+C is pressed."""
    global pool
    if pool:
        print("\n\nKeyboardInterrupt received. Terminating all download processes immediately...")
        pool.terminate() # Forcefully stop all workers
        pool.join()
    sys.exit(0)

def download_image(image_link, savefolder):
    """Downloads a single image."""
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as e:
                # NEW: Print an error message if the download fails
                print(f"Warning: Could not download {image_link} due to: {e}")
    return

def download_images(image_links, download_folder):
    """Downloads a list of images in parallel."""
    global pool # We need to modify the global pool variable

    # --- NEW: Register our emergency stop function ---
    # This tells Python to call signal_handler whenever Ctrl+C (SIGINT) is pressed.
    signal.signal(signal.SIGINT, signal_handler)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    pool = multiprocessing.Pool(4) # Assign the created pool to our global variable
    
    download_image_partial = partial(download_image, savefolder=download_folder)
    
    # Run the main download loop
    list(tqdm(pool.imap_unordered(download_image_partial, image_links), total=len(image_links)))
    
    pool.close()
    pool.join()
