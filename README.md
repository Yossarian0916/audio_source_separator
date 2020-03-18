# Audio Source Separation
Separating vocals and instruments from music clips with deep learning models

### Project python path setup
Remember to add project root absolute path (top-level directory of this project) to environment variable $PYTHONPATH

`source pythonpath.sh`

### Download DSD100 dataset

`python3 fetch_dsd100.py`

the DSD100.zip will be downloaded in data directory and unzip

### Generate tensorflow dataset

`python3 generate_dataset.py`

options: -t [stft, logstft]

default is STFT transform