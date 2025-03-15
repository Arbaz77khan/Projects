# Git Status Checker

## Overview
This Python script checks the status of multiple Git repositories inside a master folder. It identifies repositories with uncommitted changes and logs them in an Excel file (`git_status_log.xlsx`). The script appends new results to the existing file, maintaining a history of changes.

## Features
- Scans all subfolders inside a master directory for Git repositories.
- Runs `git status` to check for uncommitted changes.
- Logs the results in an Excel file with two columns:
  - **Date**: Timestamp when the script was run.
  - **Updated Folders**: List of repositories with uncommitted changes.
- Automatically creates the Excel file if it does not exist.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Git
- Required Python packages:
  ```sh
  pip install pandas openpyxl
  ```

## Usage
1. Update the script with the correct master folder path:
   ```python
   check_git_status(r"D:\Master_Folder\Data Science Course")
   ```
2. Run the script:
   ```sh
   python GitStatusChecker.py
   ```
3. Check the `git_status_log.xlsx` file for results.

## Notes
- Ensure that each subfolder is a valid Git repository.
- If a repository is clean, it will not be logged in the Excel file.
- Do not keep `git_status_log.xlsx` open while running the script to avoid write errors.

## License
This project is open-source. Feel free to modify and use it as needed.

