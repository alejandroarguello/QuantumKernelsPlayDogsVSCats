import os
import zipfile
from pathlib import Path
import gdown

def download_and_extract_dataset(dataset_option, directory_original):
    """
    Downloads and extracts the dataset based on the provided dataset_option.
    
    Args:
        dataset_option (str): Option string specifying which dataset to download.
        directory_original (str): The root directory of the project.
    
    Returns:
        tuple: (training_dir, test_dir) where the training and test data are located.
    """
    # Map dataset options to their details.
    dataset_map = {
        "Cat and Dog nº1": {
            "datasetName": "tongpython/cat-and-dog",
            "datasetZipName": "cat-and-dog.zip",
            "training_subfolder": "training_set/training_set",
            "test_subfolder": "test_set/test_set",
            "target_folder": "dataset1"
        },
        "Cat and Dog nº2": {
            "datasetName": "d4rklucif3r/cat-and-dogs",
            "datasetZipName": "cat-and-dogs.zip",
            "training_subfolder": "dataset/training_set",
            "test_subfolder": "dataset/test_set",
            "target_folder": "dataset2"
        },
        "Cat and Dog nº3": {
            "datasetName": "chetankv/dogs-cats-images",
            "datasetZipName": "dogs-cats-images.zip",
            "training_subfolder": "dataset/dataset/training_set",
            "test_subfolder": "dataset/dataset/test_set",
            "target_folder": "dataset3"
        }
    }

    if dataset_option not in dataset_map:
        raise ValueError(f"Unknown dataset option: {dataset_option}")

    info = dataset_map[dataset_option]
    datasetName = info["datasetName"]
    datasetZipName = info["datasetZipName"]
    target_folder = info["target_folder"]

    # Determine the path to the zip file
    directory_zip = os.path.join(directory_original, datasetZipName)
    
    # Download the dataset if the zip file does not exist
    if not os.path.exists(directory_zip):
        os.system(f"kaggle datasets download {datasetName}")

    # Create target directories if they do not exist.
    data_directory = os.path.join(directory_original, 'website', 'static', 'additional', 'data')
    target_path = os.path.join(data_directory, target_folder)
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)

    # If the dataset hasn't been extracted yet, extract it.
    dataset_extracted_folder = os.path.join(target_path, 'dataset')
    if not os.path.exists(dataset_extracted_folder):
        os.makedirs(dataset_extracted_folder, exist_ok=True)
        with zipfile.ZipFile(directory_zip, "r") as z:
            z.extractall(dataset_extracted_folder)
    
    # Set training and test directories based on the option.
    training_dir = os.path.join(dataset_extracted_folder, info["training_subfolder"])
    test_dir = os.path.join(dataset_extracted_folder, info["test_subfolder"])
    return training_dir, test_dir

def download_model_if_missing(model_url, target_path):
    """
    Downloads the file from model_url to target_path if it doesn't exist.
    
    Args:
        model_url (str): Google Drive shareable link or direct download URL.
        target_path (str): Local file path to save the model.
    """
    if not os.path.exists(target_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"Downloading model weights to {target_path}...")
        # gdown will handle the Google Drive download
        gdown.download(model_url, target_path, quiet=False)
        print("Download complete.")
    else:
        print("Model weights already exist.")