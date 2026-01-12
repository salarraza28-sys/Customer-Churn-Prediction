import pandas as pd
class DataLoader:
    def __init__(self, filepath):
        self.file_path=filepath
    def load_data(self):
        try:
            df=pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file not found.")
        except Exception as e:
            raise Exception(f"Erroe loading data:{e}")