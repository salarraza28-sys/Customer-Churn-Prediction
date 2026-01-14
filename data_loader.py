import pandas as pd
import logging
class DataLoader:
    def __init__(self, filepath):
        self.file_path=filepath
        self.logger = logging.getLogger(__name__)  
    def load_data(self):
        try:
            self.logger.info(f"Loading dataset from {self.file_path}")
            df=pd.read_csv(self.file_path)
            self.logger.info("Dataset loaded successfully")
            return df
        except FileNotFoundError:
            self.logger.error("Dataset file not found")
            raise FileNotFoundError("Dataset file not found.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise Exception(f"Erroe loading data:{e}")