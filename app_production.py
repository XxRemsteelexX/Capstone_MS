from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import boto3
import zipfile
import json
from botocore.exceptions import NoCredentialsError

# Real AI model imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

app = Flask(__name__)
app.secret_key = 'apollo_secret_123'

# S3 Configuration
BUCKET_NAME = "apollohealthcare-models-1753643145"
S3_CLIENT = boto3.client('s3')

# Global variables for models
text_model = None
text_tokenizer = None
image_model = None
image_metadata = None
image_transforms = None

# Medical condition mappings
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "overdose", "severe head injury", "seizure","broken","fever 104",
    "severe burns", "choking", "severe allergic reaction", "anaphylaxis",
    "fever 105", "fever 106", "fever 107", "fever 108", "high fever", "fever over 104", "104.1", 
    "104.1","104.2","104.3","104.4","104.5","104.6","104.7","104.8","104.9","105","105.1","105.2",
    "105.3","105.4","105.5","105.6","105.7","105.8","105.9","103.4","103.5","103.6","103.7","103.8",
    "103.9",
    
]

PHYSICAL_KEYWORDS = [
    "burn", "wound", "cut", "bruise", "rash", "lesion", "ulcer", "abrasion", "broken"
]

CONFIDENCE_THRESHOLD = 0.6

def download_models_from_s3():
    """Download and extract models from S3 if not present locally"""
    try:
        if not os.path.exists('./1text_model'):
            print("Downloading text model from S3...")
            S3_CLIENT.download_file(BUCKET_NAME, 'models/text_model.zip', 'text_model.zip')
            with zipfile.ZipFile('text_model.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove('text_model.zip')
            print("Text model downloaded and extracted!")
            
        if not os.path.exists('./app_models'):
            print("Downloading image model from S3...")
            S3_CLIENT.download_file(BUCKET_NAME, 'models/image_model.zip', 'image_model.zip')
            with zipfile.ZipFile('image_model.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove('image_model.zip')
            print("Image model downloaded and extracted!")
            
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

def load_text_model():
    """Load the real DistilBERT text classification model"""
    global text_model, text_tokenizer
    try:
        text_tokenizer = AutoTokenizer.from_pretrained('./1text_model')
        text_model = AutoModelForSequenceClassification.from_pretrained('./1text_model')
        text_model.eval()
        print("Real text model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading text model: {e}")
        return False

def load_image_model():
    """Load the real EfficientNet image classification model"""
    global image_model, image_metadata, image_transforms
    try:
        with open('./app_models/model_metadata.json', 'r') as f:
            image_metadata = json.load(f)
        
        image_model = timm.create_model(
            image_metadata['model_name'], 
            pretrained=False, 
            num_classes=image_metadata['num_classes']
        )
        
        checkpoint = torch.load('./app_models/best_model.pth', map_location='cpu')
        image_model.load_state_dict(checkpoint)
        image_model.eval()
        
        image_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print("Real image model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading image model: {e}")
        return False

def predict_text_symptoms(symptoms_text):
    """Real text prediction using DistilBERT model"""
    global text_model, text_tokenizer
    
    if text_model is None or text_tokenizer is None:
        return predict_text_symptoms_demo(symptoms_text)
    
    try:
        inputs = text_tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probs).item()
            predicted_class = torch.argmax(probs, dim=1).item()
            
        labels = ["ER", "Urgent Care"]
        predicted_label = labels[predicted_class]
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "predicted_class_id": predicted_class
        }
        
    except Exception as e:
        print(f"Error in text prediction: {e}")
        return predict_text_symptoms_demo(symptoms_text)

def predict_text_symptoms_demo(symptoms_text):
    """Fallback demo predictions if real model fails"""
    symptoms_lower = symptoms_text.lower()
    
    if any(word in symptoms_lower for word in ["chest pain", "heart attack", "stroke"]):
        return {"predicted_label": "ER", "confidence": 0.92, "predicted_class_id": 0}
    elif any(word in symptoms_lower for word in ["fever", "cough", "headache"]):
        return {"predicted_label": "Urgent Care", "confidence": 0.67, "predicted_class_id": 1}
    else:
        return {"predicted_label": "Urgent Care", "confidence": 0.58, "predicted_class_id": 1}

def predict_image(image_path):
    """Real image prediction using EfficientNet model"""
    global image_model, image_transforms
    
    if image_model is None or image_transforms is None:
        return predict_image_demo()
    
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = image_transforms(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            confidence = torch.max(probs).item()
            predicted_class = torch.argmax(probs, dim=1).item()
        
        class_names = ["burn_1and2", "burn_3rd", "wound_abrasions", "wound_bruises", 
                      "wound_diabetic_wounds", "wound_laceration", "wound_pressure_wounds", "wound_venous_wounds"]
        
        predicted_label = class_names[predicted_class]
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "predicted_class_id": predicted_class
        }
        
    except Exception as e:
        print(f"Error in image prediction: {e}")
        return predict_image_demo()

def predict_image_demo():
    """Fallback demo predictions for images"""
    import random
    classes = ["burn_1and2", "wound_abrasions", "burn_3rd", "wound_laceration"]
    predicted_class = random.randint(0, len(classes)-1)
    return {
        "predicted_label": classes[predicted_class],
        "confidence": round(random.uniform(0.65, 0.85), 3),
        "predicted_class_id": predicted_class
    }

# Initialize models on startup
def initialize_app():
    """Initialize the application and load models"""
    print("Initializing Apollo Healthcare Connect...")
    
    # Download models from S3 if needed
    if download_models_from_s3():
        print("Models downloaded from S3 successfully!")
        
        # Load text model
        load_text_model()
        
        # Load image model  
        load_image_model()
    else:
        print("Using demo mode - models will use fallback predictions")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()
        
        session.clear()
        session['symptoms'] = symptoms
        session['timestamp'] = datetime.now().isoformat()
        
        # Check for emergency keywords first
        if any(keyword in symptoms.lower() for keyword in EMERGENCY_KEYWORDS):
            session['emergency_detected'] = True
            session['reason'] = f"Emergency keywords detected: {symptoms}"
            return redirect(url_for('er'))
        
        # Get AI prediction
        prediction = predict_text_symptoms(symptoms)
        session['text_prediction'] = prediction
        
        # Route based on prediction and confidence
        if prediction['predicted_label'] == 'ER':
            session['reason'] = f"AI recommends emergency care (confidence: {prediction['confidence']:.1%})"
            return redirect(url_for('er'))
        elif prediction['confidence'] < CONFIDENCE_THRESHOLD:
            session['low_confidence'] = True
            return redirect(url_for('triage'))
        elif any(keyword in symptoms.lower() for keyword in PHYSICAL_KEYWORDS):
            session['physical_condition'] = True
            return redirect(url_for('upload'))
        else:
            return redirect(url_for('schedule'))
    
    return render_template("index.html")

@app.route("/triage", methods=["GET", "POST"])
def triage():
    if request.method == "POST":
        triage_info = request.form.get('more_info', '').strip()
        session['triage_info'] = triage_info
        
        # Re-analyze with additional info
        combined_symptoms = f"{session.get('symptoms', '')} {triage_info}"
        updated_prediction = predict_text_symptoms(combined_symptoms)
        session['updated_text_prediction'] = updated_prediction
        
        # Re-route based on updated analysis
        if any(keyword in combined_symptoms.lower() for keyword in EMERGENCY_KEYWORDS):
            session['reason'] = "Emergency keywords detected in additional information"
            return redirect(url_for('er'))
        elif any(keyword in combined_symptoms.lower() for keyword in PHYSICAL_KEYWORDS):
            session['physical_condition'] = True
            return redirect(url_for('upload'))
        else:
            return redirect(url_for('schedule'))
    
    return render_template("triage.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Create uploads directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            
            # Get image prediction
            image_prediction = predict_image(filepath)
            session['image_prediction'] = image_prediction
            session['uploaded_image'] = filename
            
            # Determine routing based on image analysis
            predicted_condition = image_prediction['predicted_label']
            
            # Route severe conditions to ER
            if 'burn_3rd' in predicted_condition or image_prediction['confidence'] > 0.8:
                session['reason'] = f"Severe condition detected: {predicted_condition} (confidence: {image_prediction['confidence']:.1%})"
                return redirect(url_for('er'))
            else:
                return redirect(url_for('schedule'))
    
    return render_template("upload.html")

@app.route("/er")
def er():
    reason = session.get('reason', 'Emergency care recommended based on assessment')
    return render_template("er.html", reason=reason)

@app.route("/schedule", methods=["GET", "POST"])
def schedule():
    now = datetime.now()
    
    # Check business hours (8 AM - 8 PM)
    #if now.hour < 8 or now.hour >= 20:
    #    session['reason'] = "Outside business hours - emergency care recommended"
    #    return redirect(url_for('er'))
    
    if request.method == "POST":
        session['patient_name'] = request.form.get('name', '').strip()
        session['phone'] = request.form.get('phone', '').strip()
        session['appointment_time'] = request.form.get('appointment_time', '')
        
        return redirect(url_for('confirmation'))
    
    # Generate fake appointment times (every 30 minutes from now until 8 PM)
    available_times = []
    current_time = now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0)
    if now.minute >= 30:
        current_time += timedelta(hours=1)
    
    end_time = now.replace(hour=20, minute=0, second=0, microsecond=0)
    
    while current_time <= end_time:
        available_times.append(current_time.strftime('%Y-%m-%dT%H:%M'))
        current_time += timedelta(minutes=30)

    # debug("schedule.html", available_times=available_times)
    print(f"DEBUG: Passing {len(available_times)} times to template: {available_times}")
    
    return render_template("schedule.html", available_times=available_times)

@app.route("/confirmation")
def confirmation():
    clinic_info = {
        'name': 'Apollo Healthcare Urgent Care',
        'address': '123 Medical Plaza Dr, Healthcare City, HC 12345',
        'phone': '(555) 123-CARE',
        'email': 'apollowasawonderfulcollie@aphealth.com'
    }
    return render_template("confirmation.html", clinic=clinic_info)

@app.route("/report")
def report():
    # DEBUG: Print all session data
    print("DEBUG: All session data:", dict(session))
    
    # Compile all session data for the report - FIXED VARIABLE NAMES
    report_data = {
        'patient_name': session.get('patient_name', 'N/A'),
        'patient_phone': session.get('phone', 'N/A'),  # Fixed: was 'phone'
        'appointment_time': session.get('appointment_time', 'N/A'),
        'original_symptoms': session.get('symptoms', 'N/A'),  # Fixed: was 'symptoms'
        'triage_info': session.get('triage_info', 'Not provided'),
        'text_prediction': session.get('text_prediction'),
        'updated_text_prediction': session.get('updated_text_prediction'),
        'image_prediction': session.get('image_prediction'),
        'uploaded_image': session.get('uploaded_image', 'None'),
        'timestamp': session.get('timestamp'),
        'emergency_detected': session.get('emergency_detected', False),
        'emergency_reason': session.get('reason', 'None'),  # Added this
        'low_confidence': session.get('low_confidence', False),
        'physical_condition': session.get('physical_condition', False)
    }
    
    # DEBUG: Print whats passing to template
    print("DEBUG: Report data being passed:", report_data)
    
    return render_template("report.html", data=report_data)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact") 
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    initialize_app()
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
