from torchtext.datasets import WikiText2
from torchvision.datasets import CIFAR10
from ogb.graphproppred import PygGraphPropPredDataset
import datasetcleaning as dst
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class Dataloader():
  def __init__(self):
    pass
  
  def __call__(self):
    text_transform = dst.TextDatasetTransformation()
    image_transform = dst.ImageDatasetTransformation()
    graph_transform = dst.GraphDatasetTransformation()
    
    text_subset = Subset(list(WikiText2(split='train')), range(500))
    image_subset = Subset(CIFAR10(root='./data', train=True, download=True, transform=image_transform), range(500))
    graph_subset = Subset(PygGraphPropPredDataset(name='ogbg-molhiv'), range(500))
    
    dataset = dst.MultiModalDataset(text_subset, image_subset, graph_subset,
                           text_transform, image_transform, graph_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader
