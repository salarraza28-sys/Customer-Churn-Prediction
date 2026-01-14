import logging
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        logger = logging.getLogger(__name__)
        try:
            logger.info("Evaluating model performance")
            y_pred=model.predict(X_test)
            y_prob=model.predict_proba(X_test)[:,1]
            logger.info("Accuracy:", accuracy_score(y_test,y_pred))
            logger.info("ROC-AUC:", roc_auc_score(y_test,y_prob))
            logger.info("Classification Report:\n",classification_report(y_test,y_pred))
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise Exception(f"Evaluation error : {e}")