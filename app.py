# backend/app.py

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from PIL import Image
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Global variables
model = None
animal_info = {}
is_initialized = False

def load_model_and_data():
    """Load model and animal data"""
    global model, animal_info, is_initialized
    
    try:
        # Load model
        app.logger.info("Loading MobileNetV2 model...")
        model = MobileNetV2(weights='imagenet')
        app.logger.info("Model loaded successfully")
        
        # Load animal data
        json_path = os.path.join(os.path.dirname(__file__), 'animal.json')
        app.logger.info(f"Loading animal data from {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"animal.json not found at {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process your specific JSON structure
        if isinstance(data, dict) and 'animals' in data and isinstance(data['animals'], list):
            # Create dictionary with both label and name as keys for flexibility
            animal_info = {}
            for animal in data['animals']:
                # Use label as primary key (e.g., 'goldfish')
                animal_info[animal['label'].lower()] = animal
                # Also add name as alternative key (e.g., 'goldfish (ikan mas koki)')
                animal_info[animal['name'].lower()] = animal
            
            app.logger.info(f"Loaded {len(data['animals'])} animals with {len(animal_info)} access keys")
        else:
            raise ValueError("Invalid JSON structure - expected {'animals': [...]}")
        
        # Validate we have data
        if not animal_info:
            raise ValueError("No animal data found")
            
        app.logger.info(f"Sample keys: {list(animal_info.keys())[:5]}")
        is_initialized = True
        
    except Exception as e:
        app.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        raise

@app.before_request
def initialize():
    """Initialize before first request"""
    global is_initialized
    if not is_initialized:
        load_model_and_data()

def allowed_file(filename):
    """Check allowed file extensions"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_animal_info(prediction_label):
    """Get animal info with flexible matching"""
    if not animal_info:
        return default_animal_info(prediction_label)
    
    # Generate possible variations of the label
    variations = [
        prediction_label.lower(),                      # 'goldfish'
        prediction_label.lower().replace('_', ' '),    # 'gold fish'
        prediction_label.replace('_', ' ').title(),    # 'Gold Fish'
        prediction_label.title()                       # 'Goldfish'
    ]
    
    # Check exact matches first
    for variation in variations:
        if variation in animal_info:
            return format_animal_info(animal_info[variation])
    
    # Check partial matches
    lower_label = prediction_label.lower()
    for key in animal_info.keys():
        if lower_label in key.lower():
            return format_animal_info(animal_info[key])
    
    return default_animal_info(prediction_label)

def format_animal_info(animal):
    """Format animal info consistently"""
    # Handle fun_fact which can be array or string
    fun_facts = animal.get('fun_fact', [])
    if isinstance(fun_facts, list):
        fun_fact = fun_facts[0] if fun_facts else "No fun facts available"
    else:
        fun_fact = fun_facts
    
    return {
        "name": animal.get('name', 'Unknown'),
        "description": animal.get('description', 'No description available'),
        "fun_fact": fun_fact,
        "habitat": animal.get('habitat', 'Unknown'),
        "diet": animal.get('diet', 'Unknown'),
        "size": animal.get('size', 'Unknown'),
        "conservation_status": animal.get('conservation_status', 'Unknown')
    }

def default_animal_info(label):
    """Return default info when animal not found"""
    return {
        "name": label.replace('_', ' ').title(),
        "description": f"No information available for {label.replace('_', ' ').title()}",
        "fun_fact": "No fun facts available",
        "habitat": "Unknown",
        "diet": "Unknown",
        "size": "Unknown",
        "conservation_status": "Unknown"
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if not is_initialized:
        return jsonify({'error': 'System initializing'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Process image
        image = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = preprocess_input(np.expand_dims(np.array(image), axis=0))
        
        # Make prediction
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]
        app.logger.info(f"Predictions: {decoded}")
        
        # Get best match
        _, label, confidence = decoded[0]
        info = get_animal_info(label)
        
        return jsonify({
            'label': label.replace('_', ' ').title(),
            'confidence': round(float(confidence) * 100, 2),
            'name': info['name'],
            'description': info['description'],
            'fun_fact': info['fun_fact'],
            'habitat': info['habitat'],
            'diet': info['diet'],
            'size': info['size'],
            'conservation_status': info['conservation_status']
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if is_initialized else 'initializing',
        'animals_loaded': len(animal_info),
        'model_loaded': model is not None
    })

@app.route('/animals', methods=['GET'])
def list_animals():
    """List available animals (unique by ID)"""
    unique_animals = {}
    for key, animal in animal_info.items():
        if isinstance(animal, dict) and 'id' in animal:
            unique_animals[animal['id']] = animal
    
    return jsonify({
        'animals': list(unique_animals.values()),
        'count': len(unique_animals)
    })

if __name__ == '__main__':
    try:
        load_model_and_data()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        app.logger.critical(f"Failed to start: {str(e)}")
        raise