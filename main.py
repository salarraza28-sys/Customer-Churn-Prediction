from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from predictor import ChurnPredictor
def main():
    try:
        loader=DataLoader("/Users/a.msalarraza/Downloads/Customer_Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df=loader.load_data()
        preprocessor=DataPreprocessor()
        X,y=preprocessor.preprocess(df)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        trainer=ModelTrainer("xgboost")
        model=trainer.train(X_train,y_train)
        ModelEvaluator.evaluate(model, X_test, y_test)
        predictor=ChurnPredictor(model,X.columns)
        sample=X.iloc[0].to_dict()
        prob=predictor.predict(sample)
        print(f"Churn Probability:{prob:.2f}")
    except Exception as e:
        print("Application Error:",e)
    if __name__=="__main__":
        main()