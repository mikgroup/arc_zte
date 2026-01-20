import os
import requests
from tqdm.notebook import tqdm

def download_zenodo_file(record_id, filename, access_token=None):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    print(f"Downloading {filename} from Zenodo...")

    # Handle both public and private/draft records
    url = f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
    params = {'access_token': access_token} if access_token else {}
    
    response = requests.get(url, params=params, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=filename
    ) as pbar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            f.write(data)
            pbar.update(len(data))