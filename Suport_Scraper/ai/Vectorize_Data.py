from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import pickle
import regex as re
from datetime import datetime

# Load model to GPU
model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device='cuda')

today = datetime.now()
date_str = today.strftime("%d-%m-%Y")

# Load JSON
with open(rf"Scraped_Data/products/products_{date_str}.json", 'r', encoding='utf-8') as f:
    products = json.load(f)

def clean_slug(slug: str) -> str:
    text = slug.replace("-", " ").replace("_", " ")
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prepare texts
texts = [
    f"{p['Title']} {p['categoryname']} {p['usecategory']} {clean_slug(p['main_slug'])}"
    for p in products
]

# Encode in batches (e.g., 128)
embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, device='cuda')

# Save
data = {'products': products, 'embeddings': np.array(embeddings)}
with open(r'Scraped_Data/vectors_gpu.pkl', 'wb') as f:
    pickle.dump(data, f)
