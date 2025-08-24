from flask import Flask, render_template, request, jsonify, session
import pathlib
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = "change_me_secret_key"

# Chargement du modèle
MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.before_request
def reset_session():
    if "initialized" not in session:
        session.clear()
        session["initialized"] = True


def parse_input(data):
    try:
        age = float(data.get("age", ""))
        credit_history = float(data.get("Credit_History", ""))
        loan_amount = float(data.get("LoanAmountScale", ""))
        income = float(data.get("TotalIncome", ""))
        monthly_payment = float(data.get("MonthlyPayment", ""))
    except Exception as e:
        raise ValueError(f"Entrée invalide : {e}")
    
    return {
        "age": age,
        "Credit_History": credit_history,
        "LoanAmountScale": loan_amount,
        "TotalIncome": income,
        "MonthlyPayment": monthly_payment
    }

# on fait une prédiction 
def make_prediction(input_dict):
    df_input = pd.DataFrame([input_dict])
    pred = int(model.predict(df_input)[0])
    proba = model.predict_proba(df_input)[0, 1] if hasattr(model, "predict_proba") else None
    return df_input.to_dict(orient="records")[0], pred, proba

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html", prediction=None, probability=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_dict = parse_input(request.form)
        input_vector, pred, proba = make_prediction(input_dict)
        label = "Approuvé" if pred == 1 else "Refusé"

        if pred == 1:
            return render_template("valid.html", probability=f"{proba:.3f}", input_vector=input_vector)
        else:
            return render_template("refused.html", probability=f"{proba:.3f}", input_vector=input_vector)
    except Exception as e:
        return render_template("form.html", error=str(e), prediction=None, probability=None)

@app.route("/predict_json", methods=["POST"])
def predict_json():
    try:
        data = request.get_json(silent=True) or request.form
        input_dict = parse_input(data)
        input_vector, pred, proba = make_prediction(input_dict)
        label = "Approuvé" if pred == 1 else "Refusé"
        return jsonify({
            "raw_prediction": pred,
            "prediction_label": label,
            "probability": f"{proba:.3f}" if proba is not None else None,
            "input_vector": input_vector
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
