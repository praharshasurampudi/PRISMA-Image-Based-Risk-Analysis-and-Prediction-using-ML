import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset"
)

print("Path to dataset files:", path)
