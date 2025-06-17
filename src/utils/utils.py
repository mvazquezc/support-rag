import os
import logging

class FolderDoesNotExistError(Exception):
    """Exception raised when a folder does not exist."""
    pass
class InvalidCaseFileNameError(Exception):
    """Exception raised when a case file name is invalid."""
    pass
class ChromaCollectionDeleteError(Exception):
    """Exception raised when a chroma collection couldn't be deleted."""
    pass
class Logger():
    def __init__(self, logger_name, logger_level):
        self.logger = logging.getLogger(logger_name)
        
        ch = logging.StreamHandler()
        if logger_level == "DEBUG":            
            self.logger.setLevel(logging.DEBUG)    
        elif logger_level == "ERROR":
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
    def new_logger(self):
        return self.logger

def folder_exists(folder_path):
    return os.path.isdir(folder_path)

def file_exists(file_path):
    return os.path.isfile(file_path)

def list_markdown_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.md')]