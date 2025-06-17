import os
import shutil
from datetime import datetime
import json
import argparse

def save_artifacts(source_path, artifact_type):
    """Сохраняет артефакты с версионированием"""
    base_dir = "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем структуру папок
    os.makedirs(f"{base_dir}/{artifact_type}", exist_ok=True)
    os.makedirs(f"{base_dir}/latest", exist_ok=True)
    
    if os.path.isfile(source_path):
        # Копируем с timestamp
        ext = os.path.splitext(source_path)[1]
        dest_file = f"{base_dir}/{artifact_type}/{artifact_type}_{timestamp}{ext}"
        shutil.copy2(source_path, dest_file)
        
        # Копируем как latest
        latest_file = f"{base_dir}/latest/{artifact_type}_latest{ext}"
        shutil.copy2(source_path, latest_file)
        
        print(f"Saved {source_path} to {dest_file} and {latest_file}")
    else:
        raise ValueError("Source path must be a file")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='File to save')
    parser.add_argument('--type', required=True, 
                      choices=['model', 'metrics', 'plots'], 
                      help='Type of artifact')
    args = parser.parse_args()
    
    save_artifacts(args.source, args.type)