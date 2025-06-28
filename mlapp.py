from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ✅ মডেল ও স্কেলার লোড কর
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    try:
        # ✅ ইউজারের ইনপুটগুলো float এ রূপান্তর করো
        features = [float(request.form[x]) for x in fields]

        # ✅ ফিচার স্কেল করো
        scaled_features = scaler.transform([features])

        # ✅ প্রেডিকশন
        prediction = model.predict(scaled_features)

        # ✅ রেজাল্ট লজিক ঠিক করো
        result = "Positive (Heart Disease Detected)" if prediction[0] == 1 else "Negative (No Heart Disease)"

        return f'<h2>Prediction: {result}</h2>'
    except Exception as e:
        return f'<h2>Error: {str(e)}</h2>'

if __name__ == '__main__':
    app.run(debug=True)
