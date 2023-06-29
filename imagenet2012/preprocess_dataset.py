import tensorflow_datasets as tfds

builder = tfds.builder("imagenet2012", data_dir="/rds/user/wc337/hpc-work/data")
dl_config = tfds.download.DownloadConfig(
    extract_dir="/rds/user/wc337/hpc-work/data/downloads/extracted", 
    manual_dir="/rds/user/wc337/hpc-work/data/downloads/manual",
)
builder.download_and_prepare(download_config=dl_config)