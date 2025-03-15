import os
import shutil
import time
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def backup_files(source_folder, destination_folder, log_file):
    latest_backup_folder = os.path.join(destination_folder, "Latest_Backup")
    deleted_folder = os.path.join(destination_folder, "Deleted")
    
    os.makedirs(latest_backup_folder, exist_ok=True)
    os.makedirs(deleted_folder, exist_ok=True)
    
    source_files = set()
    backup_files = set()
    
    # Collect files in source folder
    for root, _, files in os.walk(source_folder):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), source_folder)
            source_files.add(relative_path)
    
    # Collect files in latest backup
    for root, _, files in os.walk(latest_backup_folder):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), latest_backup_folder)
            backup_files.add(relative_path)
    
    with open(log_file, 'a') as log:
        # Backup new/modified files
        for file in source_files:
            source_path = os.path.join(source_folder, file)
            backup_path = os.path.join(latest_backup_folder, file)
            
            if not os.path.exists(backup_path) or os.path.getmtime(source_path) > os.path.getmtime(backup_path):
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(source_path, backup_path)
                log.write(f"[{get_timestamp()}] BACKUP: {file}\n")
        
        # Handle deleted files
        deleted_files = backup_files - source_files
        for file in deleted_files:
            prev_backup_path = os.path.join(latest_backup_folder, file)
            deleted_path = os.path.join(deleted_folder, file)
            os.makedirs(os.path.dirname(deleted_path), exist_ok=True)
            shutil.move(prev_backup_path, deleted_path)
            log.write(f"[{get_timestamp()}] DELETED: {file} moved to Deleted folder\n")

def main():
    source_folder = r"D:\Master_Folder"
    destination_folder = r"C:\Users\Arbaz Khan\OneDrive\Documents\Arbaz's Document\Master_Folder Back Up"
    log_file = os.path.join(destination_folder, "backup_log.txt")
    
    os.makedirs(destination_folder, exist_ok=True)
    
    backup_files(source_folder, destination_folder, log_file)
    print("Backup process completed.")

if __name__ == "__main__":
    main()
