import pandas as pd
from sklearn.preprocessing import StandardScaler
class DataPreprocessor:
    def __init__(self):
        self.scaler=StandardScaler
    def preprocess(self,df: pd.DataFrame):
        try:
            df=df.copy()
            df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
            df['TotalCharges']=.fillna(df['TotalCharges'].median(),inplace=True)
            df.drop('customerID',axis=1,inplace=True)
            df['Churn']=df['Churn'].map({'Yes':1,'No':0})
            categorical_cols=df.select_dtypes(include='object').columns
            df=pd.get_dummies(df,columns=categorical_cols, drop_first=True)
            X=df.drop('Churn',axis=1)
            y=df['Churn']
            return X,y 
        except KeyError as e:
            raise KeyError(f"Missing required column: {e}")
        except Exception as e:
            raise Exception(f"Preprocessing error: {e}")
    def scale(self, X_train, X_test):
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        except Exception as e:
            raise Exception(f"Scaing error: {e}")