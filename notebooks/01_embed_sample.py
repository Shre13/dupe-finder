# %% Setup
#!pip install -q transformers

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import glob
found = glob.glob("/kaggle/input/*/articles.csv") + glob.glob("/kaggle/input/*/*/articles.csv")
DATA = os.path.dirname(found[0])
print("DATA =", DATA)
N_SAMPLE = 2000          
BATCH = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# %% Load product metadata and link images
articles = pd.read_csv(f"{DATA}/articles.csv", dtype={"article_id": str})
articles["article_id"] = articles["article_id"].str.zfill(10)   

def img_path(aid):
    return f"{DATA}/images/{aid[:3]}/{aid}.jpg"

articles["img"] = articles["article_id"].map(img_path)
articles = articles[articles["img"].map(os.path.exists)]       

sample = articles.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
sample[["prod_name", "product_type_name", "colour_group_name"]].head()

# %% Attach a price (H&M prices are scaled, not dollars - fine for ranking cheaper vs pricier)
tx = pd.read_csv(f"{DATA}/transactions_train.csv",
                 usecols=["article_id", "price"], dtype={"article_id": str})
tx["article_id"] = tx["article_id"].str.zfill(10)
median_price = tx.groupby("article_id")["price"].median()
sample["price"] = sample["article_id"].map(median_price)
del tx  # free memory
sample["price"].describe()

# %% Load FashionCLIP
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device).eval()
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# %% Embed images in batches
embeddings = []
with torch.no_grad():
    for start in range(0, len(sample), BATCH):
        paths = sample["img"].iloc[start:start + BATCH]
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = model.visual_projection(vision_out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)          
        embeddings.append(feats.cpu().numpy())
        if start % 640 == 0:
            print(f"embedded {start + len(paths)}/{len(sample)}")

emb = np.vstack(embeddings).astype("float32")
print("Embedding matrix:", emb.shape)

# %% Test: find the 5 most similar items to a random product
def show_similar(query_idx, k=5):
    sims = emb @ emb[query_idx]
    top = np.argsort(-sims)[1:k + 1]                               # skip the item itself
    idxs = [query_idx] + list(top)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    for ax, i in zip(axes, idxs):
        row = sample.iloc[i]
        ax.imshow(Image.open(row["img"]))
        label = "QUERY" if i == query_idx else f"sim {sims[i]:.2f}"
        ax.set_title(f"{label}\n{row['product_type_name']}\nprice {row['price']:.4f}", fontsize=9)
        ax.axis("off")
    plt.show()

show_similar(np.random.randint(len(sample)))

# %% Save outputs (download from the notebook's Output tab)
np.save("/kaggle/working/embeddings.npy", emb)
sample.drop(columns=["img"]).to_parquet("/kaggle/working/catalog_sample.parquet")
print("Saved embeddings.npy and catalog_sample.parquet")