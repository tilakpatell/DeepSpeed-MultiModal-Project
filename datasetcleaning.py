# dataset.py (corrected)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import BertTokenizer
from torchvision import transforms

class ImageDatasetTransformation():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Corrected for 3 channels
        ])
    
    def __call__(self, image):
        return self.transform(image)

class TextDatasetTransformation():
    def __init__(self, vocab_size=10000, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.vocab_size = vocab_size
    
    def __call__(self, text):
        # Convert input to string if it's not already
        if not isinstance(text, str):
            if isinstance(text, tuple) and len(text) > 0:
                text = text[0]
            text = str(text)
            
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        tokens = [token % self.vocab_size for token in tokens]
        return torch.tensor(tokens, dtype=torch.long)

class GraphDatasetTransformation():
    def __init__(self, output_dim=128):
        self.output_dim = output_dim
        self.projection = None
        
    def __call__(self, graph):
        # Handle cases where graph might be a tuple from OGB dataset
        if isinstance(graph, tuple):
            graph = graph[0]
            
        # Extract node features
        node_features = graph.x
        
        # Handle case where there are no node features
        if node_features is None or node_features.shape[0] == 0:
            return torch.zeros(self.output_dim)
            
        # Normalize features if present
        node_features = F.normalize(node_features, p=2, dim=1)
        
        # Get graph-level representation by averaging node features
        graph_features = torch.mean(node_features, dim=0)
        
        # Ensure output dimension
        if graph_features.shape[0] != self.output_dim:
            # Apply a simple projection if dimensions don't match
            if self.projection is None or self.projection.in_features != graph_features.shape[0]:
                self.projection = torch.nn.Linear(graph_features.shape[0], self.output_dim)
            graph_features = self.projection(graph_features)
                
        return graph_features

class MultiModalDataset(Dataset):
    def __init__(self, text_data, image_data, graph_data, 
                 text_transform=None, image_transform=None, graph_transform=None):
        self.text_data = text_data
        self.image_data = image_data
        self.graph_data = graph_data
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.graph_transform = graph_transform
        
        # Ensure we use the minimum length across all datasets
        self.length = min(len(self.text_data), len(self.image_data), len(self.graph_data))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        
        # Handle image data that might be in a Subset
        if isinstance(self.image_data, Subset):
            image = self.image_data[idx][0]  # Extract the image from the (image, label) tuple
        else:
            image = self.image_data[idx]
            
        graph = self.graph_data[idx]
        
        # Apply transformations
        if self.text_transform:
            text = self.text_transform(text)
        if self.image_transform:
            # Only apply transform if it's not already transformed
            if not isinstance(image, torch.Tensor):
                image = self.image_transform(image)
        if self.graph_transform:
            graph = self.graph_transform(graph)
            
        return text, image, graph
