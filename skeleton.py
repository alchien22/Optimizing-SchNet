from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import numpy as np
import Functions as func

# 1. Data Loader for QM9 dataset
def load_data():
    dataset = QM9(root='path_to_dataset')

    # Splitting dataset into train and test
    train_dataset = dataset[:120000]
    test_dataset = dataset[120000:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

# 2. PyTorch Geometric Model
#Model Architecture:
    #Embedding
    #Gaussian smearing
    #Gaussian rbf
    #Cosine cutoff
    #for n interactions:
        #Interaction Block
            #Dense (atom-wise)
            #Filter generator
                #Dense (ssp)
                #Dense (atom-wise)
            #Continuous-filter convolution
                #Cosine modulation
                #Dense (atom-wise)
                #Propagation (message passing)
                #Dense (ssp or atom-wise)
            #Dense (ssp)
            #Dense (atom-wise)
    #Dense (atom-wise)
    #Dense (ssp)
    #Dense (atom-wise)
    #ADD: GCL
    #Sum pooling
    
class SchNetModel(nn.Module):
    def __init__(self, num_filters, num_interactions: int = 3, num_embeddings: int = 128):
        super(SchNetModel, self).__init__()
        # TODO: Define layers and modules here, for instance:
        max_z = 100
        self.num_interactions = num_interactions
        #num_embeddings = num_features = num_channels
        self.num_embeddings = num_embeddings
        self.num_filters = num_filters or num_embeddings
        
        self.embedding = nn.Embedding(max_z, self.num_embeddings, padding_idx=0) 
        #Electron densities at each node are smeared/smoothed into a continuous distribution by 
        #adding contributions from Gaussian functions at neighboring nodes
        self.smearing = func.Gaussian_RBF(num_funcs=50, cutoff=5.0)
        #Measures similarity between two nodes from 0 to 1 based on distance, 0 being not similar 
        #(further = more dissimilar)
        self.radial_basis = func.Gaussian_RBF(num_funcs=20, cutoff=5.0)
        #Gradually reduces strength of interactions between nodes as distance increases
        cutoff_instance = func.Cosine_Cutoff(5.0)
        cutoff = cutoff_instance.cutoff
        #Create interaction blocks based on the # of interactions
        self.interactions = nn.ModuleList()
        for i in range(num_interactions):
            interaction_block = func.Interaction()



    def forward(self, data):
        # TODO: Implement forward logic
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        pass

    # You might also need to implement custom `message()`, `aggregate()`, and `update()` functions

# 3. Symmetry Verification
def verify_permutation_invariance(model, data):
    # Randomly permute nodes and verify if model output remains same
    # This is a pseudocode, you'll need to implement the actual permutation
    permuted_data = data.clone() # Clone to get a new copy
    # TODO: Perform permutation on permuted_data
    return torch.allclose(model(data), model(permuted_data))

def verify_rotation_invariance(model, data):
    # Apply rotations to the graph and verify if model output remains same
    # This will require implementing a way to rotate molecular graphs.
    rotated_data = data.clone() # Clone to get a new copy
    # TODO: Apply rotation on rotated_data
    return torch.allclose(model(data), model(rotated_data))

# 4. Evaluation
def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data)
            # Choose a target, for example, the 0th property
            target = data.y[:, 0]
            loss = F.mse_loss(out, target)
            total_loss += loss.item()
    return total_loss / len(loader)

# Main program
def main():
    train_loader, test_loader = load_data()
    
    model = SchNetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Sample training loop
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y[:, 0])  # Using the 0th property as an example target
            loss.backward()
            optimizer.step()
        
        # Evaluate
        test_loss = evaluate_model(model, test_loader)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
    
    # Symmetry verification on a sample batch
    # sample_data = next(iter(test_loader))
    # print("Permutation Invariance:", verify_permutation_invariance(model, sample_data))
    # print("Rotation Invariance:", verify_rotation_invariance(model, sample_data))

if __name__ == '__main__':
    main()
