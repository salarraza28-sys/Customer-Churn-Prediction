from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
class ModelTrainer:
    def __init__(self, model_type:str):
        self.model_type=model_type
        self.model=self.__initialize__model()
    def __initialize__model(self):
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=42)
        elif self.model_type == "xgboost":
            return XGBClassifier(n_estimators=200, learning_rate=0.05, max_dept=5, eval_metric="logloss")
        else:
            raise ValueError(f"Unsupported Model type")
    def train(self, X, y):
        try:
            self.model.fit(X,y)
            return self.model
        except Exception as e:
            raise Exception(f"Model Training failed: {e}") 