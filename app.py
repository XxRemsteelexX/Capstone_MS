from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'apollo_secret_123'  # Replace with a secure key

# Load model
model_path = "./1text_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["cold/flu", "allergy", "covid", "migraine", "burn"]
EMERGENCY_LABELS = ["heart attack", "stroke", "severe trauma"]  # Optional to match later
PHYSICAL_LABELS = ["burn", "rash"]  # Add any physical ones here

CONFIDENCE_THRESHOLD = 0.6  # Below this, we ask for more info

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probs).item()
            predicted_class = torch.argmax(probs, dim=1).item()
            label = labels[predicted_class]

        # Store in session
        session.clear()
        session['symptoms'] = symptoms
        session['prediction'] = label
        session['confidence'] = confidence

        if label in EMERGENCY_LABELS:
            return redirect(url_for('er'))
        elif label in PHYSICAL_LABELS:
            return redirect(url_for('upload'))
        elif confidence < CONFIDENCE_THRESHOLD:
            return redirect(url_for('triage'))
        else:
            return redirect(url_for('schedule'))

    return render_template("index.html")

@app.route("/triage", methods=["GET", "POST"])
def triage():
    if request.method == "POST":
        session['triage_info'] = request.form['more_info']
        return redirect(url_for('upload'))  # Could be upload or ER, use logic here later
    return render_template("triage.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Placeholder: accept image and run model here
        session['image_result'] = 'moderate burn'
        return redirect(url_for('schedule'))
    return render_template("upload.html")

@app.route("/er")
def er():
    return render_template("er.html")

@app.route("/schedule", methods=["GET", "POST"])
def schedule():
    now = datetime.now()
    hour = now.hour
    if hour < 8 or hour >= 18:
        return redirect(url_for('er'))

    if request.method == "POST":
        session['patient_name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['appt_time'] = request.form['time']
        return redirect(url_for('confirmation'))

    return render_template("schedule.html", now=now.strftime('%Y-%m-%dT%H:%M'))

@app.route("/confirmation")
def confirmation():
    return render_template("confirmation.html")

@app.route("/report")
def report():
    return render_template("report.html", session=session)

if __name__ == "__main__":
    app.run(debug=True)

