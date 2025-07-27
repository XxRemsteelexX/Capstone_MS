from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from PIL import Image
import json
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime, timedelta
import timm
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.secret_key = 'apollo_secret_123'  # Replace with a secure key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image Model Architecture (matching your training script)
class AdvancedMedicalClassifier(nn.Module):
    """Advanced medical image classifier with multiple architectures"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=8, dropout_rate=0.3, pretrained=True):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Advanced pooling and attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 8, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        avg_features = self.global_avg_pool(features).view(features.size(0), -1)
        max_features = self.global_max_pool(features).view(features.size(0), -1)
        attention_weights = self.attention(avg_features)
        attended_features = avg_features * attention_weights
        combined_features = torch.cat([attended_features, max_features], dim=1)
        output = self.classifier(combined_features)
        return output

# Load Text Classification Model
print("Loading text classification model...")
text_model_path = "./1text_model"  # Your DistilBERT model path
try:
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path)
    text_model.eval()
    
    # Load label mappings if available
    try:
        with open(os.path.join(text_model_path, "label_mappings.json"), 'r') as f:
            label_mappings = json.load(f)
            text_id2label = {int(k): v for k, v in label_mappings["id2label"].items()}
            text_label2id = label_mappings["label2id"]
    except FileNotFoundError:
        # Default mappings from your training
        text_id2label = {0: "ER", 1: "Urgent Care"}
        text_label2id = {"ER": 0, "Urgent Care": 1}
    
    print("Text model loaded successfully!")
    print(f"Text model classes: {text_id2label}")
    
except Exception as e:
    print(f"Error loading text model: {e}")
    text_model = None
    text_tokenizer = None

# Load Image Classification Model
print("Loading image classification model...")
image_model_path = "./app_models"  # Path from your image training script
try:
    # Load model metadata
    with open(os.path.join(image_model_path, "model_metadata.json"), 'r') as f:
        image_metadata = json.load(f)
    
    # Initialize model
    image_model = AdvancedMedicalClassifier(
        model_name=image_metadata['model_name'],
        num_classes=image_metadata['num_classes'],
        dropout_rate=image_metadata['dropout_rate'],
        pretrained=False
    )
    
    # Load trained weights
    image_model.load_state_dict(torch.load(
        os.path.join(image_model_path, "best_model.pth"),
        map_location='cpu'
    ))
    image_model.eval()
    
    image_class_names = image_metadata['class_names']
    image_idx_to_class = image_metadata['idx_to_class']
    
    print("Image model loaded successfully!")
    print(f"Image model classes: {image_class_names}")
    
except Exception as e:
    print(f"Error loading image model: {e}")
    image_model = None
    image_class_names = []

# Image preprocessing function
def get_image_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

# Configuration for business logic
CONFIDENCE_THRESHOLD = 0.6  # Below this,  ask for more info
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "overdose", "severe head injury", "seizure",
    "severe burns", "choking", "severe allergic reaction", "anaphylaxis"
]

PHYSICAL_KEYWORDS = [
    "burn", "wound", "cut", "bruise", "swelling", "rash", "skin", "injury",
    "fracture", "broken", "sprain", "bite", "sting", "laceration"
]

# Emergency image classifications that should go to ER
ER_IMAGE_CONDITIONS = [
    "severe_burn", "deep_wound", "fracture", "severe_injury", "gunshot", "stab_wound"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_text_symptoms(symptoms_text):
    """Predict using text model"""
    if text_model is None or text_tokenizer is None:
        return {"predicted_label": "Urgent Care", "confidence": 0.5, "error": "Text model not loaded"}
    
    try:
        inputs = text_tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probs).item()
            predicted_class = torch.argmax(probs, dim=1).item()
            label = text_id2label[predicted_class]
        
        return {
            "predicted_label": label,
            "confidence": confidence,
            "predicted_class_id": predicted_class
        }
    except Exception as e:
        print(f"Error in text prediction: {e}")
        return {"predicted_label": "Urgent Care", "confidence": 0.5, "error": str(e)}

def predict_image(image_path):
    """Predict using image model"""
    if image_model is None:
        return {"predicted_label": "minor_condition", "confidence": 0.5, "error": "Image model not loaded"}
    
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transforms = get_image_transforms()
        augmented = transforms(image=image)
        image_tensor = augmented['image'].unsqueeze(0)
        
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            confidence = torch.max(probs).item()
            predicted_class = torch.argmax(probs, dim=1).item()
            label = image_class_names[predicted_class]
        
        return {
            "predicted_label": label,
            "confidence": confidence,
            "predicted_class_id": predicted_class
        }
    except Exception as e:
        print(f"Error in image prediction: {e}")
        return {"predicted_label": "minor_condition", "confidence": 0.5, "error": str(e)}

def contains_emergency_keywords(text):
    """Check if text contains emergency keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EMERGENCY_KEYWORDS)

def contains_physical_keywords(text):
    """Check if text suggests a physical condition"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in PHYSICAL_KEYWORDS)

def generate_fake_appointments(current_time):
    """Generate fake appointment times"""
    appointments = []
    
    # Business hours: 8 AM to 8 PM
    start_hour = 8
    end_hour = 20
    
    # If it's after hours, show next day appointments
    if current_time.hour >= end_hour or current_time.hour < start_hour:
        next_day = current_time + timedelta(days=1)
        base_date = next_day.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        is_next_day = True
    else:
        # Show appointments for today, starting from next hour
        next_hour = current_time.hour + 1
        if next_hour >= end_hour:
            # Too late today, show tomorrow
            next_day = current_time + timedelta(days=1)
            base_date = next_day.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            is_next_day = True
        else:
            base_date = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            is_next_day = False
    
    # Generate appointments every 30 minutes
    current_slot = base_date
    end_time = base_date.replace(hour=end_hour)
    
    while current_slot < end_time:
        appointments.append({
            'time': current_slot.strftime('%Y-%m-%dT%H:%M'),
            'display': current_slot.strftime('%A, %B %d at %I:%M %p')
        })
        current_slot += timedelta(minutes=30)
    
    return appointments, is_next_day

def get_nearest_ers():
    """Return fake ER locations for demo"""
    return [
        {
            'name': 'Mercy Hospital Emergency Room',
            'address': '1234 Medical Center Dr, Waterloo, IA 50701',
            'phone': '(319) 555-0123',
            'distance': '2.1 miles'
        },
        {
            'name': 'UnityPoint Health Emergency Department',
            'address': '5678 Health Plaza, Cedar Falls, IA 50613',
            'phone': '(319) 555-0456',
            'distance': '5.7 miles'
        }
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()
        
        if not symptoms:
            return render_template("index.html", error="Please enter your symptoms")
        
        # Clear previous session data
        session.clear()
        
        # Store original symptoms
        session['original_symptoms'] = symptoms
        session['timestamp'] = datetime.now().isoformat()
        
        # Predict using text model
        prediction = predict_text_symptoms(symptoms)
        session['text_prediction'] = prediction
        
        # Check for emergency keywords first
        if contains_emergency_keywords(symptoms):
            session['emergency_reason'] = "Emergency keywords detected"
            return redirect(url_for('er'))
        
        # Check prediction result
        if prediction['predicted_label'] == 'ER':
            session['emergency_reason'] = "Model predicted emergency care needed"
            return redirect(url_for('er'))
        
        # Check confidence level
        if prediction['confidence'] < CONFIDENCE_THRESHOLD:
            session['low_confidence'] = True
            return redirect(url_for('triage'))
        
        # Check if physical condition
        if contains_physical_keywords(symptoms):
            session['physical_condition'] = True
            return redirect(url_for('upload'))
        
        # Otherwise, proceed to scheduling
        return redirect(url_for('schedule'))
    
    return render_template("index.html")

@app.route("/triage", methods=["GET", "POST"])
def triage():
    if request.method == "POST":
        additional_info = request.form.get("details", "").strip()
        session['triage_info'] = additional_info
        
        # Combine original symptoms with additional info
        combined_text = f"{session.get('original_symptoms', '')} {additional_info}"
        
        # Re-run prediction with more info
        new_prediction = predict_text_symptoms(combined_text)
        session['updated_text_prediction'] = new_prediction
        
        # Check again for emergency
        if contains_emergency_keywords(combined_text) or new_prediction['predicted_label'] == 'ER':
            session['emergency_reason'] = "Emergency detected after additional information"
            return redirect(url_for('er'))
        
        # Check if physical condition mentioned
        if contains_physical_keywords(combined_text):
            session['physical_condition'] = True
            return redirect(url_for('upload'))
        
        # Otherwise proceed to scheduling
        return redirect(url_for('schedule'))
    
    return render_template("triage.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("upload.html", error="No image file selected")
        
        file = request.files['image']
        if file.filename == '':
            return render_template("upload.html", error="No image file selected")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict using image model
            image_prediction = predict_image(filepath)
            session['image_prediction'] = image_prediction
            session['uploaded_image'] = filename
            
            # Check if image shows emergency condition
            if image_prediction['predicted_label'] in ER_IMAGE_CONDITIONS:
                session['emergency_reason'] = f"Image shows emergency condition: {image_prediction['predicted_label']}"
                return redirect(url_for('er'))
            
            # Proceed to scheduling
            return redirect(url_for('schedule'))
        else:
            return render_template("upload.html", error="Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)")
    
    return render_template("upload.html")

@app.route("/er")
def er():
    ers = get_nearest_ers()
    return render_template("er.html", 
                         emergency_rooms=ers,
                         reason=session.get('emergency_reason', 'Emergency care recommended'))

@app.route("/schedule", methods=["GET", "POST"])
def schedule():
    current_time = datetime.now()
    
    # Check if after hours (before 8 AM or after 8 PM)
    if current_time.hour >= 20 or current_time.hour < 8:
        return redirect(url_for('er'))
    
    if request.method == "POST":
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        appointment_time = request.form.get('appointment_time', '')
        
        if not all([name, phone, appointment_time]):
            appointments, is_next_day = generate_fake_appointments(current_time)
            return render_template("schedule.html", 
                                 appointments=appointments,
                                 is_next_day=is_next_day,
                                 error="Please fill in all fields")
        
        # Store appointment info
        session['patient_name'] = name
        session['patient_phone'] = phone
        session['appointment_time'] = appointment_time
        session['appointment_booked'] = True
        
        return redirect(url_for('confirmation'))
    
    # Generate appointment times
    appointments, is_next_day = generate_fake_appointments(current_time)
    
    return render_template("schedule.html", 
                         appointments=appointments,
                         is_next_day=is_next_day)

@app.route("/confirmation")
def confirmation():
    if not session.get('appointment_booked'):
        return redirect(url_for('index'))
    
    # Parse appointment time for display
    apt_time = session.get('appointment_time', '')
    if apt_time:
        try:
            apt_datetime = datetime.fromisoformat(apt_time)
            formatted_time = apt_datetime.strftime('%A, %B %d, %Y at %I:%M %p')
        except:
            formatted_time = apt_time
    else:
        formatted_time = "Not specified"
    
    clinic_info = {
        'name': 'Apollo Urgent Care Center',
        'address': '789 Healthcare Blvd, Waterloo, IA 50701',
        'phone': '(319) 555-CARE (2273)'
    }
    
    return render_template("confirmation.html", 
                         patient_name=session.get('patient_name'),
                         appointment_time=formatted_time,
                         clinic=clinic_info)

@app.route("/report")
def report():
    # Compile all session data for the report
    report_data = {
        'timestamp': session.get('timestamp'),
        'patient_name': session.get('patient_name', 'Not provided'),
        'patient_phone': session.get('patient_phone', 'Not provided'),
        'appointment_time': session.get('appointment_time', 'Not scheduled'),
        'original_symptoms': session.get('original_symptoms', 'Not provided'),
        'triage_info': session.get('triage_info', 'Not provided'),
        'text_prediction': session.get('text_prediction', {}),
        'updated_text_prediction': session.get('updated_text_prediction', {}),
        'image_prediction': session.get('image_prediction', {}),
        'uploaded_image': session.get('uploaded_image', 'None'),
        'emergency_reason': session.get('emergency_reason', 'None'),
        'physical_condition': session.get('physical_condition', False),
        'low_confidence': session.get('low_confidence', False)
    }
    
    return render_template("report.html", data=report_data)

# Additional static pages
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# API endpoint for getting appointment times (if needed for dynamic updates)
@app.route("/api/appointments")
def api_appointments():
    current_time = datetime.now()
    appointments, is_next_day = generate_fake_appointments(current_time)
    return jsonify({
        'appointments': appointments,
        'is_next_day': is_next_day
    })

if __name__ == "__main__":
    app.run(debug=True)