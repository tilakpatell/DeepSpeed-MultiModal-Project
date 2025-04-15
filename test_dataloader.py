import torch
import os
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import IMDB
from torchvision.datasets import CIFAR10
from ogb.graphproppred import PygGraphPropPredDataset
import datasetcleaning as dst

# Create transformations first
text_transform = dst.TextDatasetTransformation()
image_transform = dst.ImageDatasetTransformation()
graph_transform = dst.GraphDatasetTransformation()

# Set a stable download directory in the current project for ghe datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load test datasets with proper transformations
test_wiki_data = []
try:
    for i, (label, text) in enumerate(IMDB(split='test', root=DATA_DIR)):
        if i >= 100:
            break
        test_wiki_data.append(text)
except Exception as e:
    print(f"Warning: IMDB test dataset loading error: {e}")
    # Fallback to a small set of dummy text samples if loading fails
    test_wiki_data = ["Sample test movie review text"] * 100

# Use the image_transform
test_image_subset = Subset(CIFAR10(root=DATA_DIR, train=False, download=True, 
                             transform=image_transform), range(100))

# Handle the special indexing
graph_dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=DATA_DIR)
test_indices = range(len(graph_dataset) - 100, len(graph_dataset))
test_graph_subset = []
for i in test_indices:
    test_graph_subset.append(graph_dataset[i])

test_dataset = dst.MultiModalDataset(
    test_wiki_data, 
    test_image_subset, 
    test_graph_subset,
    text_transform, 
    image_transform, 
    graph_transform
)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
