# segment.py

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models.segmentation as seg
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = seg.deeplabv3_resnet101(weights=seg.DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def segment_image(pil_img):
    img = pil_img.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)["out"]
        mask = output.argmax(1).squeeze().cpu().numpy()

    # Highlight class 15 (person) or make a colored mask
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_mask[mask == 15] = [0, 255, 0]  # Green overlay

    img_np = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)

    return Image.fromarray(overlay)
