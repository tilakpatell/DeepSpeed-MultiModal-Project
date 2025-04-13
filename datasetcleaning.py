from torch.utils.data import Subset
from transformers import BertTokenizer
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.vocab_size = vocab_size
    
    def __call__(self, text):
       tokens = self.tokenizer.encode(
          text,
          add_special_tokens=True,
          max_length = self.max_length,
          truncation=True,
          padding='max_length'
       )
       tokens = [token % self.vocab_size for token in tokens]
       return torch.tensor(tokens, dtype=torch.long)
    

class GraphDatasetTransformation():
    def __init__(self, output_dim=128):
        self.output_dim = output_dim
    
    def __call__(self, graph):
      try:
        node_features = graph.x
        if node_features is not None:
            node_features = F.normalize(node_features, p=2, dim=1)
        graph_features = torch.mean(node_features, dim=0)
        return graph_features
      except Exception as e:
        print(f"There was the following exception: {e}, using a zero vector instead. \n")
        return torch.zeros(self.output_dim)

class MultiModalDataset(Dataset):
    def __init__(self, text_data, image_data, graph_data, 
                 text_transform=None, image_transform=None, graph_transform=None):
        self.text_data = text_data
        self.image_data = image_data
        self.graph_data = graph_data
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.graph_transform = graph_transform
        self.length = min(len(text_data), len(image_data), len(graph_data))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.text_data[idx][0]
        image = self.image_data[idx][0]
        graph = self.graph_data[idx]
        
        if self.text_transform:
            text = self.text_transform(text)
        if self.image_transform:
            image = self.image_transform(image)
        if self.graph_transform:
            graph = self.graph_transform(graph)
            
        return text, image, graph
