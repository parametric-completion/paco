import os
import requests

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def download(cfg: DictConfig):
    # Define the target directory and URL
    target_dir = "./ckpt"
    url = cfg.checkpoint_url

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get the filename from the Content-Disposition header
    response = requests.head(url, allow_redirects=True)
    content_disposition = response.headers.get("Content-Disposition", "")
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[1].strip('";')
    else:
        filename = "ckpt-best.pth"  # Fallback filename if header is missing

    # Full path for the file
    target_file = os.path.join(target_dir, filename)

    # Download the file
    print(f"Downloading to {target_file}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error if the request fails
    with open(target_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {filename} to {target_dir}")


if __name__ == "__main__":
    download()
