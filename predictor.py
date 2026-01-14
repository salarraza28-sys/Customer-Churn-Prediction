import logging
import pandas as pd
class ChurnPredictor:
    def __init__(self, model,feature_columns):
        self.model=model
        self.feature_columns=feature_columns
        self.logger = logging.getLogger(__name__)
    def predict(self, customer_data:dict):
        try:
            self.logger.info("Starting churn prediction")   
            df=pd.DataFrame([customer_data])
            df=pd.get_dummies(df)
            df=df.reindex(columns=self.feature_columns,fill_value=0)
            churn_prob=self.model.predict_proba(df)[0][1]
            self.logger.info(f"Churn prediction successful: {churn_prob:.2f}")
            return churn_prob
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise Exception(f"Prediction error: {e}")