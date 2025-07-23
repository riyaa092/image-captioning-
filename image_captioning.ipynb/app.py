import torch
from caption_model import Captioner  # your model class
from utils import Vocabulary          # your vocab class
import pickle
import os
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
import numpy as np
import cv2
from torchvision import models, transforms
import torchvision

# ========== Model Setup ========== #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load Captioning model
model = Captioner()
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device).eval()

# ResNet feature extractor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modules = list(resnet.children())[:-1]
resnet = torch.nn.Sequential(*modules).to(device).eval()

# Load DeepLabV3 for segmentation
segmentation_model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
segmentation_model = segmentation_model.to(device).eval()

# ========== Flask Setup ========== #

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========== Caption Generation ========== #

def generate_caption(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(image_tensor).squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(0)  # [1, 2048]
        caption = model.generate_caption(features, vocab)

    return caption

# ========== Image Segmentation ========== #

def segment_image(image, save_path):
    image = image.convert("RGB").resize((224, 224))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = segmentation_model(image_tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 15] = [0, 255, 0]

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(image_cv, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(save_path, blended)

# ========== Routes ========== #

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file and uploaded_file.filename != '':
            original_filename = uploaded_file.filename
            original_path = os.path.join(UPLOAD_FOLDER, original_filename)
            uploaded_file.save(original_path)

            img = Image.open(original_path)

            # Generate caption
            caption = generate_caption(img)

            # Segment and save output
            segmented_filename = original_filename.rsplit('.', 1)[0] + '_segmented.jpg'
            segmented_path = os.path.join(OUTPUT_FOLDER, segmented_filename)
            segment_image(img, segmented_path)

            # Generate static URLs using url_for
            return render_template('index.html',
                                   caption=caption,
                                   original_path=url_for('uploaded_file', filename=original_filename),
                                   segmented_path=url_for('segmented_file', filename=segmented_filename))
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/outputs/<filename>')
def segmented_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# ========== Run the App ========== #

if __name__ == '__main__':
    app.run(debug=True)
