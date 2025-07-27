import os
import pandas as pd
import numpy as np
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

class SymptomClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the Symptom Classifier
        
        Args:
            model_path (str): Path to saved model directory. If None, will train new model.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label2id = {"ER": 0, "Urgent Care": 1}
        self.id2label = {0: "ER", 1: "Urgent Care"}
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def load_data(self, csv_path):
        """Load and prepare the dataset"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Create label mappings
        unique_labels = sorted(df['label'].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        print(f"Label mapping: {self.label2id}")
        
        # Apply label encoding
        df['label_id'] = df['label'].map(self.label2id)
        
        return df
    
    def prepare_datasets(self, df, test_size=0.2, random_state=42):
        """Split and prepare datasets for training"""
        # Split data with stratification
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        print(f"Train set size: {len(df_train)}")
        print(f"Test set size: {len(df_test)}")
        
        # Prepare datasets
        train_data = df_train[['text', 'label_id']].copy()
        test_data = df_test[['text', 'label_id']].copy()
        
        # Rename for transformers
        train_data = train_data.rename(columns={'label_id': 'labels'})
        test_data = test_data.rename(columns={'label_id': 'labels'})
        
        # Convert to HuggingFace datasets
        dataset_train = Dataset.from_pandas(train_data)
        dataset_test = Dataset.from_pandas(test_data)
        
        return dataset_train, dataset_test
    
    def setup_tokenizer(self, model_name="distilbert-base-uncased"):
        """Initialize and setup tokenizer"""
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )
    
    def prepare_model(self, model_name="distilbert-base-uncased"):
        """Load and prepare the model"""
        print(f"Loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
    
    def train(self, csv_path, output_dir="./symptom_classifier_model", 
              epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Train the symptom classifier
        
        Args:
            csv_path (str): Path to training CSV file
            output_dir (str): Directory to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
        """
        # Load and prepare data
        df = self.load_data(csv_path)
        dataset_train, dataset_test = self.prepare_datasets(df)
        
        # Setup tokenizer and model
        self.setup_tokenizer()
        self.prepare_model()
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        dataset_train = dataset_train.map(self.tokenize_function, batched=True)
        dataset_test = dataset_test.map(self.tokenize_function, batched=True)
        
        # Remove text column
        dataset_train = dataset_train.remove_columns(['text'])
        dataset_test = dataset_test.remove_columns(['text'])
        
        # Load accuracy metric
        accuracy_metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy_metric.compute(predictions=predictions, references=labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            save_total_limit=2,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=compute_metrics,
            processing_class=self.tokenizer,
        )
        
        print("Starting training...")
        try:
            # Train the model
            train_result = trainer.train()
            
            print(f"Training completed!")
            print(f"Training loss: {train_result.training_loss:.4f}")
            
            # Evaluate the model
            print("Evaluating model...")
            eval_result = trainer.evaluate()
            
            print("Evaluation results:")
            for key, value in eval_result.items():
                print(f"  {key}: {value:.4f}")
            
            # Save the model
            print("Saving model...")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save label mappings
            with open(f"{output_dir}/label_mappings.json", "w") as f:
                json.dump({
                    "label2id": self.label2id,
                    "id2label": self.id2label
                }, f, indent=2)
            
            self.model_path = output_dir
            print("Model saved successfully!")
            
            return eval_result
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None
    
    def load_model(self):
        """Load a pre-trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")
        
        print(f"Loading model from {self.model_path}")
        
        # Load label mappings if available
        mappings_path = os.path.join(self.model_path, "label_mappings.json")
        if os.path.exists(mappings_path):
            with open(mappings_path, "r") as f:
                mappings = json.load(f)
                self.label2id = mappings["label2id"]
                self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, text, return_all_scores=False):
        """
        Predict the classification for given text
        
        Args:
            text (str): Input text to classify
            return_all_scores (bool): Whether to return all class probabilities
            
        Returns:
            dict: Prediction results with label, confidence, and optionally all scores
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probabilities).item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        result = {
            "predicted_label": self.id2label[predicted_class],
            "confidence": confidence,
            "predicted_class_id": predicted_class
        }
        
        if return_all_scores:
            result["all_scores"] = {
                self.id2label[i]: score.item() 
                for i, score in enumerate(probabilities[0])
            }
        
        return result
    
    def evaluate_model(self, csv_path):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        df = self.load_data(csv_path)
        predictions = []
        true_labels = []