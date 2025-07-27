from flask import Flask, render_template, request
import joblib  # or torch, etc.

app = Flask(__name__)

# Load your model
model = joblib.load('models/text_model.pkl')  # or your actual loading code

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        # Process and predict
        prediction = model.predict([symptoms])[0]  # depends on your model
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
