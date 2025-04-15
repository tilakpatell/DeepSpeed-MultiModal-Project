import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms

class ImageDatasetTransformation():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __call__(self, image):
        return self.transform(image)

class TextDatasetTransformation():
    def __init__(self, vocab_size=10000, max_length=128):
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def __call__(self, text):
        if not isinstance(text, str):
            if isinstance(text, tuple) and len(text) > 0:
                text = text[0]
            text = str(text)
            
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        tokens = tokens[:self.max_length]
        
        tokens = tokens + [''] * (self.max_length - len(tokens))
        
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        
        return torch.tensor(token_ids, dtype=torch.long)

class GraphDatasetTransformation():
    def __init__(self, output_dim=128):
        self.output_dim = output_dim
        self.projection = None
        
    def __call__(self, graph):
        if isinstance(graph, tuple):
            graph = graph[0]
            
        node_features = graph.x
        
        if node_features is None or node_features.shape[0] == 0:
            return torch.zeros(self.output_dim)
            
        if node_features.dtype == torch.long or node_features.dtype == torch.int:
            node_features = node_features.float()
            
        node_features = F.normalize(node_features, p=2, dim=1).detach()
        
        graph_features = torch.mean(node_features, dim=0)
        
        if graph_features.shape[0] != self.output_dim:
            if self.projection is None or self.projection.in_features != graph_features.shape[0]:
                self.projection = torch.nn.Linear(graph_features.shape[0], self.output_dim)
            graph_features = self.projection(graph_features)
                
        return graph_features.detach()

class MultiModalDataset(Dataset):
    def __init__(self, text_data, image_data, graph_data, 
                 text_transform=None, image_transform=None, graph_transform=None):
        self.text_data = text_data
        self.image_data = image_data
        self.graph_data = graph_data
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.graph_transform = graph_transform
        
        self.length = min(len(self.text_data), len(self.image_data), len(self.graph_data))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        
        if isinstance(self.image_data, Subset):
            image = self.image_data[idx][0]
        else:
            image = self.image_data[idx]
            
        graph = self.graph_data[idx]
        
        if self.text_transform:
            text = self.text_transform(text)
        if self.image_transform:
            if not isinstance(image, torch.Tensor):
                image = self.image_transform(image)
        if self.graph_transform:
            graph = self.graph_transform(graph)
            
        return text, image, graph
