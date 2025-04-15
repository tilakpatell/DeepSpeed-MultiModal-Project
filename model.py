import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, dim_feedforward=hidden_dim, nhead=4), num_layers=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)  
        x = self.encoder(x)
        x = x.permute(1, 2, 0)  
        return self.pool(x).squeeze(-1) 
    
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        target_dtype = self.conv[0].weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return self.conv(x)

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        # Convert input to match weight dtype if needed
        if x.dtype != self.fc1.weight.dtype:
            x = x.to(self.fc1.weight.dtype)
            
        x = F.relu(self.fc1(x))
        return self.fc2(x)   

class FusionModel(nn.Module):
    def __init__(self, vocab_size=10000, text_dim=64, graph_in=128):
        super().__init__()
        
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=text_dim)
        self.image_encoder = ImageEncoder()
        self.graph_encoder = GraphEncoder(in_dim=graph_in)
        fusion_input_dim = text_dim + 32 + 32 

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, text, image, graph):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        
        if graph.dtype != self.graph_encoder.fc1.weight.dtype:
            graph = graph.to(dtype=self.graph_encoder.fc1.weight.dtype)
            
        graph_features = self.graph_encoder(graph)
        
        if text_features.dtype != image_features.dtype:
            text_features = text_features.to(image_features.dtype)
        if graph_features.dtype != image_features.dtype:
            graph_features = graph_features.to(image_features.dtype)
        
        combined_features = torch.cat([text_features, image_features, graph_features], dim=1)
        
        return self.fusion(combined_features)
