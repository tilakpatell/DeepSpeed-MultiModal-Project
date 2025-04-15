# test_dataloader.py
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

# Load test datasets with proper transformations
# For WikiText2, collect samples into a list first
test_wiki_data = []
for i, text in enumerate(WikiText2(split='test')):
    if i >= 100:  # Using smaller test set
        break
    test_wiki_data.append(text)

# For CIFAR10, use the image_transform
test_image_subset = Subset(CIFAR10(root='./data', train=False, download=True, 
                             transform=image_transform), range(100))

# For graph data, handle the special indexing
graph_dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
# Use the last 100 samples for testing
test_indices = range(len(graph_dataset) - 100, len(graph_dataset))
test_graph_subset = []
for i in test_indices:
    test_graph_subset.append(graph_dataset[i])

# Create the multimodal test dataset
test_dataset = dst.MultiModalDataset(
    test_wiki_data, 
    test_image_subset, 
    test_graph_subset,
    text_transform, 
    image_transform, 
    graph_transform
)

# Create test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
