from flask import Flask, render_template, request
import joblib
from model import train_model

app = Flask(__name__)

# Train the model and load it
train_model()
model = joblib.load('result_predictor.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get form data
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous_marks'])
        assignments = float(request.form['assignments'])
        study_hours = float(request.form['study_hours'])

        # Predict
        input_features = [[attendance, previous_marks, assignments, study_hours]]
        result = model.predict(input_features)[0]
        prediction = "Pass ✅" if result == 1 else "Fail ❌"

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
