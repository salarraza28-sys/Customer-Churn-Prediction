import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
class ModelTrainer:
    def __init__(self, model_type:str):
        self.logger = logging.getLogger(__name__)
        self.model_type=model_type
        self.model=self.__initialize__model()
    def __initialize__model(self):
        self.logger.info(f"Initializing model: {self.model_type}")
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=42)
        elif self.model_type == "xgboost":
            return XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss")
        else:
            self.logger.error("Unsupported model type provided")
            raise ValueError(f"Unsupported Model type")
    def train(self, X, y):
        try:
            self.logger.info("Starting model training")
            self.model.fit(X,y)
            self.logger.info("Model training completed successfully")
            return self.model
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise Exception(f"Model Training failed: {e}") 