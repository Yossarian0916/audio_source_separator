import os
import requests
from zipfile import ZipFile


# make data directory if not exists
data_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

save_path = os.path.join(data_dir, "DSD100.zip")
print("save path: ", save_path)

# download chunk by chunk, decrease memory usage
url = "http://liutkus.net/DSD100.zip"
response = requests.get(url, allow_redirects=True, stream=True)
with open(save_path, 'wb') as fd:
    print('Download DSD100 dataset...')
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            fd.write(chunk)
print('Download Done!')

with ZipFile(save_path, 'r') as zip:
    print('Extracting files...')
    zip.extractall()
    print('Done!')
