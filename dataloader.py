import torch
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import WikiText2
from torchvision.datasets import CIFAR10
from ogb.graphproppred import PygGraphPropPredDataset
import datasetcleaning as dst

# Create transformations first
text_transform = dst.TextDatasetTransformation()
image_transform = dst.ImageDatasetTransformation()
graph_transform = dst.GraphDatasetTransformation()

# Load datasets with proper transformations
# For WikiText2, collect samples into a list first
wiki_data = []
for i, text in enumerate(WikiText2(split='train')):
    if i >= 500:
        break
    wiki_data.append(text)

# For CIFAR10, use the image_transform
image_subset = Subset(CIFAR10(root='./data', train=True, download=True), range(500))

# For graph data, handle the special indexing
graph_dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
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

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
