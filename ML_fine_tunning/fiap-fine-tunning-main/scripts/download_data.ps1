# Define the directory paths
$dataDir = "../data/celeba"
$zipFile = "$dataDir\celeba-dataset.zip"
$extractDir = "$dataDir\img_align_celeba"

# Create the data directory if it doesn't exist
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir
    Write-Host "Created directory: $dataDir"
}

# Navigate to the data directory
Set-Location $dataDir

# Download the dataset using Kaggle API
Write-Host "Downloading CelebA dataset from Kaggle..."
kaggle datasets download -d smnishanth/celeba-dataset -p $dataDir

# Check if the dataset was downloaded successfully
if (Test-Path $zipFile) {
    Write-Host "Download completed: $zipFile"
} else {
    Write-Host "Download failed!"
    exit 1
}

# Unzip the dataset
Write-Host "Extracting the dataset..."
Expand-Archive -Path $zipFile -DestinationPath $extractDir

# Check if the extraction was successful
if (Test-Path $extractDir) {
    Write-Host "Extraction completed: $extractDir"
} else {
    Write-Host "Extraction failed!"
    exit 1
}

# Remove the zip file to clean up
Write-Host "Cleaning up..."
Remove-Item $zipFile

Write-Host "Dataset download and extraction completed successfully!"
