import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
class DataPreprocessor:
    def __init__(self):
        self.scaler=StandardScaler
        self.logger = logging.getLogger(__name__)
    def preprocess(self,df: pd.DataFrame):
        try:
            self.logger.info("Starting data preprocessing")
            df=df.copy()
            df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
            # df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
            df['TotalCharges'] = df['TotalCharges'].apply(lambda x: np.random.choice(df['TotalCharges'].dropna()) if pd.isna(x) else x)
            df.drop('customerID',axis=1,inplace=True)
            df['Churn']=df['Churn'].map({'Yes':1,'No':0})
            categorical_cols=df.select_dtypes(include='object').columns
            df=pd.get_dummies(df,columns=categorical_cols, drop_first=True)
            X=df.drop('Churn',axis=1)
            y=df['Churn']
            self.logger.info("Data preprocessing completed successfully")
            return X,y 
        except KeyError as e:
            self.logger.error(f"Missing required column: {e}")
            raise KeyError(f"Missing required column: {e}")
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise Exception(f"Preprocessing error: {e}")
    def scale(self, X_train, X_test):
        try:
            self.logger.info("Scaling features")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.logger.info("Feature scaling completed")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            self.logger.error(f"Scaling error: {e}")
            raise Exception(f"Scaing error: {e}")