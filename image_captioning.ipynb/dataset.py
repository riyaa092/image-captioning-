# dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import pickle

class CaptionDataset(Dataset):
    def __init__(self, caption_file, image_folder, vocab_path, transform=None):
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        with open(caption_file, "r") as f:
            self.data = [line.strip().split("|") for line in f if "|" in line]

        self.image_folder = image_folder
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = caption.lower().strip().split()
        caption_idx = [self.vocab.word2idx["<start>"]]
        caption_idx += [self.vocab.word2idx.get(word, self.vocab.word2idx["<unk>"]) for word in tokens]
        caption_idx.append(self.vocab.word2idx["<end>"])

        return image, torch.tensor(caption_idx)
