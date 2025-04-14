import os
import requests
import zipfile

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def download_and_unzip(cfg: DictConfig):
    # Define directories and URL
    target_dir = "./data"
    url = cfg.dataset_url  # OneDrive URL for abc.zip from config
    filename = "abc.zip"

    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Full path for the temporary zip file
    temp_file = os.path.join(target_dir, filename)

    # Download the file
    print(f"Downloading {filename} to {temp_file}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error if the request fails
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {filename} to {temp_file}")
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return

    # Unzip the file
    print(f"Unzipping {filename} to {target_dir}...")
    try:
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Unzipped {filename} to {target_dir}")
    except zipfile.BadZipFile as e:
        print(f"Unzip failed: {e}")
        return
    finally:
        # Clean up the temporary zip file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Removed temporary file {temp_file}")


if __name__ == "__main__":
    download_and_unzip()
