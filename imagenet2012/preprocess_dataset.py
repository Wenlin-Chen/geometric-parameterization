import tensorflow_datasets as tfds

builder = tfds.builder("imagenet2012", data_dir="data")
dl_config = tfds.download.DownloadConfig(
    extract_dir="data/downloads/extracted", 
    manual_dir="data/downloads/manual",
)
builder.download_and_prepare(download_config=dl_config)