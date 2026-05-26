"""
Dataset Download Script
Downloads the Wine Quality dataset from UCI ML Repository
"""

import urllib.request
import os

def download_wine_dataset():
    """Download wine quality dataset from UCI ML Repository"""
    
    os.makedirs("dataset", exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    output_path = os.path.join("dataset", "winequality-red.csv")
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return output_path
    
    print("Downloading Wine Quality dataset from UCI ML Repository...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Dataset downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_wine_dataset()
