from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Load the trained model and PCA transformer
model = joblib.load("lymphoma_model1.joblib")
pca = joblib.load("pca_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🎯 Collect 30 PCA feature values from HTML form
        features = [float(request.form[f'feature{i}']) for i in range(1, 31)]
        input_data = np.array(features).reshape(1, -1)

        # 🧬 Predict Lymphoma Type
        prediction = model.predict(input_data)
        result = prediction[0].decode() if isinstance(prediction[0], bytes) else prediction[0]

        # 📊 Get prediction probabilities for all lymphoma classes
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0].tolist()
            class_labels = model.classes_.tolist()
        else:
            probabilities = None
            class_labels = None

        return render_template(
            'index.html',
            prediction_text=f"🧬 Predicted Lymphoma Type: {result}",
            probabilities=probabilities,
            class_labels=class_labels
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

# ✅ Entry point
if __name__ == "__main__":
    app.run(debug=True)
