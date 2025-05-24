import os
import subprocess
import pandas as pd
from datetime import datetime

def check_git_status(master_folder, output_file="git_status_log.xlsx"):
    updated_folders = []
    
    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # List all subfolders in the master folder
    for folder in os.listdir(master_folder):
        folder_path = os.path.join(master_folder, folder)
        
        # Check if it's a directory and has a .git folder (i.e., a git repo)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, '.git')):
            try:
                # Run 'git status' command in git bash
                result = subprocess.run(['git', '-C', folder_path, 'status'], capture_output=True, text=True, check=True)
                
                # Check if output contains 'nothing to commit, working tree clean'
                if "nothing to commit, working tree clean" not in result.stdout:
                    updated_folders.append(folder)
            except subprocess.CalledProcessError as e:
                print(f"Error checking {folder}: {e}")
                continue
    
    # Prepare data for the Excel file
    if not updated_folders:
        updated_folders.append("No Updates")
    
    new_data = pd.DataFrame({"Date": [current_date] * len(updated_folders), "Updated Folders": updated_folders})
    
    # Append to the existing Excel file or create a new one
    try:
        existing_data = pd.read_excel(output_file)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data
    
    
    print(f"Saving file at: {os.path.abspath(output_file)}")
    updated_data.to_excel(output_file, index=False)
    print(f"Check complete. Results saved in {output_file}")

output_file = r"D:\Master_Folder\Data Science Course\Projects\Git status checker\git_status_log.xlsx"
check_git_status(r"D:\Master_Folder\Data Science Course", output_file)

