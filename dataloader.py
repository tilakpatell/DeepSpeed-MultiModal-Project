import torch
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import IMDB
from torchvision.datasets import CIFAR10
from ogb.graphproppred import PygGraphPropPredDataset
import datasetcleaning as dst
import os


text_transform = dst.TextDatasetTransformation()
image_transform = dst.ImageDatasetTransformation()
graph_transform = dst.GraphDatasetTransformation()


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


wiki_data = []
try:
    for i, (label, text) in enumerate(IMDB(split='train', root=DATA_DIR)):
        if i >= 500:
            break
        wiki_data.append(text)
except Exception as e:
    print(f"Warning: IMDB dataset loading error: {e}")
    
    wiki_data = ["Sample movie review text"] * 500


image_subset = Subset(CIFAR10(root=DATA_DIR, train=True, download=True), range(500))


graph_dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=DATA_DIR)
graph_subset = []
for i in range(min(500, len(graph_dataset))):
    graph_subset.append(graph_dataset[i])


dataset = dst.MultiModalDataset(
    wiki_data, 
    image_subset, 
    graph_subset,
    text_transform, 
    image_transform, 
    graph_transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
