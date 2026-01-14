from flask import Flask, render_template, request
import joblib
from predictor import ChurnPredictor

app = Flask(__name__)

# Load model and features
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

predictor = ChurnPredictor(model, feature_columns)

@app.route("/", methods=["GET", "POST"])
def index():
    churn_prob = None

    if request.method == "POST":
        customer_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
            "Contract": request.form["Contract"]
        }

        churn_prob = round(predictor.predict(customer_data), 2)

    return render_template("index.html", churn_prob=churn_prob)


if __name__ == "__main__":
    app.run(debug=True)
