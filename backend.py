from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from safetensors.torch import load_file
from PIL import Image
import io
import os

# --- Configuration ---
MODEL_PATH = 'pneumonia_model_resnet.safetensors'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Use an absolute path to find the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Initialize Flask App ---
# Serve static files (like index.html) from the script's directory
app = Flask(__name__, static_folder=SCRIPT_DIR)
CORS(app)

# --- Model Definition & Loading ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
device = torch.device("cpu")

model_full_path = os.path.join(SCRIPT_DIR, MODEL_PATH)
if not os.path.exists(model_full_path):
    print(f"FATAL ERROR: Model file not found at {model_full_path}")
    model = None
else:
    try:
        state_dict = load_file(model_full_path, device=device.type)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        model = None

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- API Endpoints ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    
    # --- FIX 1: Expect the key 'file' to match the new JavaScript ---
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided with key "file"'}), 400

    file = request.files['file']
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        # --- FIX 2: Return JSON with keys the JavaScript expects ---
        return jsonify({
            'normalProbability': confidences.get('NORMAL', 0),
            'pneumoniaProbability': confidences.get('PNEUMONIA', 0)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process the image'}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Your app will be available at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

