# Apollo Healthcare Connect

**A Production-Grade Multi-Modal Deep Learning Medical Triage System**

*Western Governors University Data Science Capstone Project*  
*Submitted by: Glenn Dalbey | August 2025*

---

##  **Live Production System**
**https://apollohealthcareconnect.com**

*Serving real users worldwide with intelligent healthcare routing and provider preparation*

---

##  **Project Overview**

Apollo Healthcare Connect addresses critical inefficiencies in healthcare appointment booking by providing **intelligent patient routing** between urgent care and emergency room facilities while preparing healthcare providers with comprehensive patient information prior to appointments.

### ** Research Question**
*"Can a comprehensive multi-modal artificial intelligence system be developed and successfully deployed as a live production application to provide accurate healthcare routing while maintaining clinical safety standards?"*

### ** Answer: YES** - Proven with live deployment and exceptional performance metrics.

---

##  **Key Achievements**

| Metric | Result | Significance |
|--------|---------|-------------|
| **Combined Multi-Modal Accuracy** | **93.8%** | Exceeds clinical implementation thresholds |
| **Burn Classification Accuracy** | **98.0%** | Critical for emergency routing decisions |
| **Text Classification Accuracy** | **94.0%** | DistilBERT symptom analysis |
| **Dataset Scale** | **8,085 medical images** | Across 8 medical conditions |
| **Class Imbalance Handled** | **29.7:1 ratio** | Advanced techniques (focal loss, label smoothing) |
| **Production Deployment** | ** Live System** | Real-world validation |

---

##  **Technical Innovation**

### **Multi-Modal AI Architecture**
- **Text Analysis:** DistilBERT-based symptom classifier (Urgent Care vs ER)
- **Image Analysis:** Sophisticated 5-model ensemble for medical image classification
- **Advanced PyTorch Model:** 8-class wound/burn classifier with extreme class imbalance handling

### **Ensemble Methodology**
- **5 Deep Learning Models:** EfficientNetB0/B1, ResNet50, DenseNet121 variants
- **Weighted Consensus:** Medical-optimized thresholds (0.35) for clinical safety
- **Advanced Loss Functions:** Focal loss (α=1, γ=2) + label smoothing (0.1)
- **Conservative Routing:** Prioritizes patient safety with uncertainty quantification

### **Production-Ready Features**
- **Real-Time Inference:** Sub-second response times
- **Scalable Architecture:** Flask + AWS S3 + cloud deployment
- **Safety Protocols:** Conservative thresholds prevent unsafe routing
- **Provider Preparation:** Automated patient prep materials

---

##  **Dataset & Methodology**

### **Medical Image Dataset**
- **Sources:** 4 Kaggle medical datasets (burns, wounds)
- **Total Images:** 8,085 across 8 medical conditions
- **Classes:** burn_1and2 (4,876), burn_3rd (1,023), wound types (varying sizes)
- **Challenge:** Extreme class imbalance (29.7:1 ratio) successfully handled

### **Text Dataset**
- **Synthetic Generation:** 250 balanced symptom descriptions
- **Categories:** Urgent Care vs Emergency Room triage decisions
- **Processing:** DistilBERT tokenization with medical context preservation

### **Advanced Techniques**
- **Class Imbalance Mitigation:** Oversampling + focal loss + label smoothing
- **Data Augmentation:** Medical-specific 15-step pipeline
- **Multi-Source Integration:** Unified preprocessing across diverse datasets
- **Production Optimization:** Model compression for real-time inference

---

##  **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │ -> │  Flask Backend   │ -> │  AI Ensemble    │
│  (HTML/CSS/JS)  │    │ (Production App) │    │ (Multi-Modal)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
               ┌───────▼────────┐ ┌─────▼──────┐
               │ Text Classifier │ │Image Models│
               │   (DistilBERT)  │ │(5-Ensemble)│
               └────────────────┘ └────────────┘
```

---

##  **Repository Structure**

```
Apollo-Healthcare-Connect/
├── app.py                       # Development server
├── app_production.py            # Production deployment
├── requirements.txt             # Dependencies
├── templates/                   # Web interface templates
├── static/                      # Frontend assets
├── burn_ensemble_models/        # 5-model ensemble weights
├── pytorch_model_outputs/       # 8-class PyTorch model
├── data/                        # Sample datasets
├── utils/                       # Preprocessing utilities
└── README.md                    # This file
```

---

##  **Quick Start**

### **Installation**
```bash
git clone https://github.com/your-username/apollo-healthcare-connect
cd apollo-healthcare-connect
pip install -r requirements.txt
```

### **Local Development**
```bash
python app.py
# Visit: http://localhost:5000
```

### **Production Deployment**
```bash
gunicorn --bind 0.0.0.0:5000 app_production:app
```

---

##  **Requirements**

### **Core Dependencies**
- **Flask** - Web framework
- **PyTorch** - Deep learning (8-class model)
- **TensorFlow/Keras** - Ensemble models
- **Transformers** - DistilBERT (Hugging Face)
- **OpenCV** - Image processing
- **Albumentations** - Medical image augmentation
- **Boto3** - AWS S3 integration

### **System Requirements**
- **Python 3.8+**
- **GPU recommended** (for inference speed)
- **8GB+ RAM** (for model loading)

---

## ⚠ **Important Disclaimers**

- **Educational Purpose:** This system was developed as a data science capstone project
- **Not Medical Advice:** Results are for demonstration and research purposes only
- **Clinical Validation Required:** Professional medical evaluation needed before clinical use
- **Research Only:** This system is not approved for diagnostic or treatment decisions

---

##  **Academic Context**

### **Capstone Significance**
This project demonstrates advanced data science methodology including:
- Multi-modal AI system development
- Extreme class imbalance handling
- Production deployment and MLOps
- Healthcare workflow integration
- Research-to-production pipeline

### **Technical Contributions**
- Novel ensemble methodology for medical image classification
- Successful handling of 29.7:1 class imbalance
- Real-world deployment validation
- Conservative safety-first routing protocols

---

##  **Performance Metrics**

### **Model Performance**
- **Burn Classification:** 98% accuracy (critical for emergency routing)
- **Text Analysis:** 94% accuracy (symptom understanding)
- **Combined System:** 93.8% multi-modal accuracy
- **Processing Speed:** 50 seconds for 1,227 images (production hardware)

### **Production Metrics**
- **Response Time:** Sub-second inference
- **Uptime:** 99%+ availability
- **Safety Record:** Conservative routing prevents inappropriate ER-level urgent care bookings

---

##  **Research Applications**

This system provides a foundation for:
- Healthcare workflow optimization research
- Multi-modal AI development
- Medical image classification advancement
- Production ML deployment methodologies


---

##  **Acknowledgements**

- **Western Governors University** - Data Science Program support
- **Hugging Face** - DistilBERT transformer models
- **TensorFlow/PyTorch Communities** - Deep learning frameworks
- **Kaggle** - Medical image datasets for research
- **Healthcare AI Research Community** - Inspiration and methodology guidance

---

##  **Contact**

**Glenn Dalbey**  
*Data Science Graduate, Western Governors University*  
*Capstone Project Completed: August 2025*

**Live System:** https://apollohealthcareconnect.com

---

*This project represents the culmination of advanced data science education, demonstrating the practical application of machine learning techniques to real-world healthcare challenges.*
