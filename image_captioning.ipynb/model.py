# model.py
import torch
from caption_model import Captioner
import torchvision.models as models
import torchvision.transforms as T
import pickle
from utils import Vocabulary
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load model
model = Captioner()
model.load_state_dict(torch.load("caption_model.pth", map_location=device))
model = model.to(device).eval()

# Feature extractor
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modules = list(resnet.children())[:-1]
resnet = torch.nn.Sequential(*modules).to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def generate_caption(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(tensor).squeeze()
        caption = model.generate_caption(features.unsqueeze(0), vocab)
    return caption
