import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

# Load structured dataset
df = pd.read_csv("data/structured_data.csv")

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model.eval()

# Image preprocessing (must match ImageNet training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze().numpy()  # shape: (2048,)

embeddings = []

print("Extracting embeddings...")

for idx, row in df.iterrows():
    img_path = row["image_path"]

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    emb = extract_embedding(img_path)
    embeddings.append(emb)

    if idx % 200 == 0:
        print(f"Processed {idx} images")

# Convert embeddings to DataFrame
emb_df = pd.DataFrame(embeddings)

# Save embeddings
os.makedirs("embeddings", exist_ok=True)
emb_df.to_csv("embeddings/image_embeddings.csv", index=False)

print("\nImage embeddings extracted and saved")
print("Embedding shape:", emb_df.shape)
