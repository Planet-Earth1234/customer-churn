import os
import torch
from torch import nn
from modell import CustomEfficientNet, li  # Assuming CustomEfficientNet is your model class
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from torchvision import transforms

# Configure the Gemini API
# Initialize Flask app
app = Flask(__name__)
cors = CORS(app, origins="*")

# Check if CUDA is available and set device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained EfficientNet model for image classification
model_img = CustomEfficientNet()

# Load model weights on the correct device
model_weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')
model_img.load_state_dict(torch.load(model_weights_path, map_location=device))

# Move the model to the selected device (either CPU or GPU)
model_img.to(device)
model_img.eval()

# Preprocessing for images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image classification API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load the image
        image = Image.open(io.BytesIO(file.read()))
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device (GPU/CPU)

        # Make prediction
        with torch.no_grad():
            output = model_img(image_tensor)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)

        # Return the predicted class
        nums = predicted_class.item()
        # You can also modify this if your model also provides raw material data
        return jsonify({'predicted_class': li[nums], 'raw_materials': "Some raw materials here"})

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})






# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5174)
