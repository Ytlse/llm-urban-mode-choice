"""
archive_log.py - Log archiving script.

This script is meant to be run only once at service stack startup
to archive the existing log file before new logs are written.

It reads the configuration (services/llm-agents/config/config.yaml) to find the path
of the log file.
"""
import os
from datetime import datetime
from settings import settings

if hasattr(settings, 'app') and hasattr(settings.app, 'log_file') and settings.app.log_file:
    log_file_path = str(settings.app.log_file)
    if os.path.exists(log_file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{log_file_path}.{timestamp}.bak"
        try:
            os.rename(log_file_path, archive_path)
            print(f"INFO: Archived existing log file to {archive_path}")
        except OSError as e:
            print(f"ERROR: Failed to archive log file '{log_file_path}': {e}")