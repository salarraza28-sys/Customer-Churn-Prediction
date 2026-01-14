import pandas as pd
class ChurnPredictor:
    def __init__(self, model,feature_columns):
        self.model=model
        self.feature_columns=feature_columns
    def predict(self, customer_data:dict):
        try:
            df=pd.DataFrame([customer_data])
            df=pd.get_dummies(df)
            df=df.reindex(columns=self.feature_columns,fill_value=0)
            churn_prob=self=self.model.predict_proa(df)[0][1]
            return churn_prob
        except Exception as e:
            raise Exception(f"Prediction error: {e}")