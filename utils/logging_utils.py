import logging
import os
import time
import psutil
import torch
from datetime import datetime


def setup_logger(client_id):
    """Set up logging for a specific client."""
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root (two levels up from utils)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Create logs directory in the project root
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'client_{client_id}')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler with absolute path
    log_file = os.path.join(logs_dir, f'client_{client_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add both handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_metrics(logger, metrics_dict):
    """Log model metrics (accuracy, loss) and memory usage."""
    # Log metrics
    for metric_name, value in metrics_dict.items():
        logger.info(f"{metric_name}: {value}")
    
    # Log memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory Usage - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info(f"GPU Memory Usage: {gpu_memory:.2f} MB")

class CommunicationTimer:
    """Context manager to measure communication time."""
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"Communication time for {self.operation_name}: {elapsed_time:.4f} seconds") 