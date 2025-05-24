# Automatic Backup System

## Overview
The **Automatic Backup System** is a Python script designed to back up files and folders from a specified source location to a backup destination. The system ensures that:
- Initially, a full backup is created.
- Subsequent backups only copy **new or modified** files.
- Deleted files from the source are moved to a separate **'Deleted'** folder in the backup.
- A log file maintains a record of backup activities.
- The script runs automatically every **24 hours** using Windows Task Scheduler.

## Features
✅ **Full Initial Backup** - Copies all files from the source folder initially.  
✅ **Incremental Backup** - Only copies files that are new or modified.  
✅ **Deleted File Handling** - Moves deleted files to a dedicated folder instead of removing them.  
✅ **Logging System** - Tracks all backup operations in a log file.  
✅ **Automated Execution** - Uses Windows Task Scheduler to run every 24 hours.  

## Installation & Setup
### 1️⃣ **Prerequisites**
- Windows OS
- Python installed (Check with `python --version` in the command prompt)

### 2️⃣ **Download the Script**
Save the `backup_script.py` file to your desired location.

### 3️⃣ **Modify Paths in the Script**
Open `backup_script.py` and update the following variables to match your setup:
```python
source_folder = r"D:\Master_Folder"
destination_folder = r"C:\Users\Arbaz Khan\OneDrive\Documents\Arbaz's Document\Master_Folder Back Up"
```

### 4️⃣ **Run the Script Manually (First Time)**
To verify that the script works:
```sh
python "C:\Users\Arbaz Khan\backup_script.py"
```
Check the `Master_Folder Back Up` directory to see if the backup was created.

### 5️⃣ **Set Up Windows Task Scheduler (Automate the Script)**
1. Open **Task Scheduler** (`Win + R` → `taskschd.msc` → Enter).
2. Click **Create Basic Task...**
3. Name the task: **Daily Backup Task**
4. Select **Daily** and set the preferred time.
5. Choose **Start a Program** as the action.
6. In **Program/script**, enter:
   ```
   python
   ```
7. In **Add arguments**, enter:
   ```
   "C:\Users\Arbaz Khan\backup_script.py"
   ```
8. Click **Finish** and manually test it by right-clicking the task and selecting **Run**.

## How It Works
- The script scans `Master_Folder` for files.
- It compares them with the latest backup in `Master_Folder Back Up/Latest_Backup`.
- **New or modified files** are copied to `Latest_Backup`.
- **Deleted files** from the source are moved to `Deleted` inside the backup folder.
- The `backup_log.txt` records all changes.

## Logs
A log file is maintained in the backup folder:
```
[2025-03-16_03-34-58] BACKUP: report.pdf
[2025-03-17_03-34-58] DELETED: old_document.docx moved to Deleted folder
```

## Notes
- The script does **not** compress files.
- No automatic deletion of old backups (for now).
- Ensure Python is correctly configured in Task Scheduler.

## Troubleshooting
- **Task not running?** Check Task Scheduler history and ensure the script path is correct.
- **Files not updating?** Ensure `python` is in the system PATH and check the logs for errors.
- **Deleted files missing?** They should be in `Deleted` inside the backup folder.

## Future Enhancements
🔹 Option to **compress** old backups.  
🔹 Set a **retention period** for backups.  
🔹 GUI version for easier configuration.  

## License
This project is open-source and free to use.

