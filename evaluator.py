from sklearn.metrics import accuracy_score, classification_report, roc_curve
class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        try:
            y_pred=model.predict(X_test)
            y_prob=model.predict_proba(X_test)[:,1]
            print("Accuracy:",accuracy_score(y_test,y_pred))
            print("ROC-AUC:",accuracy_score(y_test,y_prob))
            print("Classification Report:\n",classification_report(y_test,y_pred))
        except Exception as e:
            raise Exception(f"Evaluation error : {e}")