import torch
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import IMDB
from torchvision.datasets import CIFAR10
from ogb.graphproppred import PygGraphPropPredDataset
import datasetcleaning as dst
import os

# Create transformations first
text_transform = dst.TextDatasetTransformation()
image_transform = dst.ImageDatasetTransformation()
graph_transform = dst.GraphDatasetTransformation()

# Set a stable download directory in the current project
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load datasets with proper transformations
# For IMDB, collect samples into a list first
wiki_data = []
try:
    for i, (label, text) in enumerate(IMDB(split='train', root=DATA_DIR)):
        if i >= 500:
            break
        wiki_data.append(text)
except Exception as e:
    print(f"Warning: IMDB dataset loading error: {e}")
    # Fallback to a small set of dummy text samples if loading fails
    wiki_data = ["Sample movie review text"] * 500

# For CIFAR10, use the image_transform
image_subset = Subset(CIFAR10(root=DATA_DIR, train=True, download=True), range(500))

# For graph data, handle the special indexing
graph_dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=DATA_DIR)
graph_subset = []
for i in range(min(500, len(graph_dataset))):
    graph_subset.append(graph_dataset[i])

# Create the multimodal dataset
dataset = dst.MultiModalDataset(
    wiki_data, 
    image_subset, 
    graph_subset,
    text_transform, 
    image_transform, 
    graph_transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
