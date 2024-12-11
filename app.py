import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

# Model Architecture (Same as before)
class PotholeDetector(nn.Module):
    def __init__(self):
        super(PotholeDetector, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

# Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Logging Configuration
def setup_logging(app):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler(
        'logs/pothole_api.log', 
        maxBytes=10240, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Pothole Detection API startup')

# Model Loading
def load_model():
    device = torch.device("cpu")  # Use CPU for free hosting compatibility
    model = PotholeDetector()
    
    # Check if model file exists
    model_path = 'best_pothole_detector.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Image Preparation
def prepare_image(image_base64):
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logging.error(f"Image preparation error: {str(e)}")
        raise

# Flask App Configuration
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Setup Logging
setup_logging(app)

# Load Model Globally
try:
    model, device = load_model()
except Exception as e:
    app.logger.error(f"Model loading failed: {str(e)}")
    model, device = None, None

# Rate Limiting (Basic Implementation)
request_count = {}
MAX_REQUESTS_PER_IP = 50  # Adjust as needed

def rate_limit(ip):
    current_count = request_count.get(ip, 0)
    if current_count >= MAX_REQUESTS_PER_IP:
        return False
    request_count[ip] = current_count + 1
    return True

@app.route('/predict', methods=['POST'])
def predict():
    # Get client IP
    ip = request.remote_addr

    # Basic Rate Limiting
    if not rate_limit(ip):
        return jsonify({
            'error': 'Rate limit exceeded. Try again later.',
            'status': 429
        }), 429

    # Validate Model Loaded
    if model is None:
        return jsonify({
            'error': 'Model not properly initialized',
            'status': 500
        }), 500

    try:
        # Get base64 encoded image
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image provided',
                'status': 400
            }), 400
        
        # Prepare and predict
        image_tensor = prepare_image(data['image']).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted.item()].item() * 100
        
        # Log successful prediction
        app.logger.info(f"Prediction: {'Pothole' if predicted.item() == 1 else 'Plain Road'}")
        
        return jsonify({
            'class': 'Pothole' if predicted.item() == 1 else 'Plain Road',
            'confidence': round(confidence, 2),
            'status': 200
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'status': 500
        }), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'status': 404
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'status': 500
    }), 500

# For Local Development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))