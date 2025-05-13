import os
import requests

WEIGHTS = {
    'transunet_classifier.pth': 'https://huggingface.co/<username>/AI-Powered-Radiology-Assistant/resolve/main/checkpoints/transunet_classifier.pth',
    'unet_segmentation.h5':    'https://huggingface.co/<username>/AI-Powered-Radiology-Assistant/resolve/main/checkpoints/unet_segmentation.h5'
}

CHECKPOINT_DIR = 'checkpoints'

def download_file(filename: str, url: str):
    dest_path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(dest_path):
        print(f\"{filename} already exists, skipping.\")
        return
    print(f\"Downloading {filename}...\")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f\"Downloaded {filename} to {dest_path}.\")

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for fname, url in WEIGHTS.items():
        try:
            download_file(fname, url)
        except Exception as e:
            print(f\"Failed to download {fname}: {e}\")

if __name__ == '__main__':
    main()
