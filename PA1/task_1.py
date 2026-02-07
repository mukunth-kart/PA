##Building GloVe
from utils import Vocab
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

def build_cooccurrence_matrix(file_path, vocab, window_size=5):
    """
    Builds a co-occurrence matrix by skipping OOV words and 
    using a sparse-friendly dictionary to save RAM.
    """
    # Using a dictionary to store counts: {(id_i, id_j): count}
    cooc_counts = defaultdict(float)
    
    print(f"Reading corpus from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            # 1. Tokenize and convert to IDs
            # The 'if' check here handles your requirement to skip words not in vocab
            tokens = line.strip().split()
            ids = []
            for w in tokens:
                word_id = vocab.get_id(w)
                if word_id is not None:
                    ids.append(word_id)
            
            # 2. Sliding Window logic
            for i, target_id in enumerate(ids):
                # Define window boundaries
                start = max(0, i - window_size)
                end = min(len(ids), i + window_size + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    
                    context_id = ids[j]
                    # We only store one direction if we want a symmetric matrix, 
                    # but usually, we store the specific (i, j) relationship.
                    cooc_counts[(target_id, context_id)] += 1.0

            # if line_idx % 10000 == 0:
        print(f"Processed All lines...")

    # 3. Convert dictionary to a Torch Sparse Tensor
    return dict_to_sparse_tensor(cooc_counts, len(vocab.word2id))

def dict_to_sparse_tensor(counts_dict, vocab_size):
    """
    Converts the dictionary of counts into a PyTorch Sparse COO tensor.
    """
    if not counts_dict:
        return torch.sparse_coo_tensor(size=(vocab_size, vocab_size))

    # Extract indices and values
    indices = torch.tensor(list(counts_dict.keys())).t()  # Shape: [2, num_non_zero]
    values = torch.tensor(list(counts_dict.values()), dtype=torch.float32)
    
    return torch.sparse_coo_tensor(indices, values, (vocab_size, vocab_size))

# def weight_loss(preds, labels, x_max, indices, X):
#     if X[indices[0] ,indices[1]] > x_max:
#         return torch.ones(size=indices.shape[0])
#     return 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        
        # Layer 1: input_dim -> hidden_dim
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Layer 2: hidden_dim -> hidden_dim
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Layer 3 (Output): hidden_dim -> hidden_dim
        # Note: No BatchNorm or Dropout here as per your request
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(GloVe, self).__init__()
        
        # padding_idx=0 ensures <UNK> stays as a zero vector and is ignored
        self.w_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.c_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.w_biases = nn.Embedding(vocab_size, 1, padding_idx=0)
        self.c_biases = nn.Embedding(vocab_size, 1, padding_idx=0)

    def forward(self, target_ids, context_ids):
        w_vec = self.w_embeddings(target_ids)
        c_vec = self.c_embeddings(context_ids)
        
        w_bias = self.w_biases(target_ids).squeeze()
        c_bias = self.c_biases(context_ids).squeeze()
        
        dot_product = (w_vec * c_vec).sum(dim=1)
        return dot_product + w_bias + c_bias
# Usage Example:
# model = EmbeddingModel(vocab_size=len(vocab.word2id), embed_dim=300)

class GloVeDataset(Dataset):
    def __init__(self, sparse_matrix):
        """
        sparse_matrix: A torch.sparse_coo_tensor
        """
        # Get the indices (i, j) and values (counts)
        # indices shape: [2, nnz], values shape: [nnz]
        self.indices = sparse_matrix.coalesce().indices()
        self.values = sparse_matrix.coalesce().values()
        
        # Filter out cases where i == j
        mask = self.indices[0] != self.indices[1]
        self.indices = self.indices[:, mask]
        self.values = self.values[mask]
        
        self.num_samples = self.values.size(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Returns: target_id, context_id, cooccurrence_count
        return self.indices[0, idx], self.indices[1, idx], self.values[idx]
    

class GloVeLoss(nn.Module):
    def __init__(self, x_max=100, alpha=0.75):
        super(GloVeLoss, self).__init__()
        self.x_max = x_max
        self.alpha = alpha

    def forward(self, predictions, x_ij):
        """
        predictions: Output from the EmbeddingModel (dot product + biases)
        x_ij: The actual co-occurrence counts from the dataset
        """
        # 1. Compute the weight for each sample
        # If x_ij > x_max, weight is 1.0. Otherwise, (x_ij/x_max)^alpha
        weights = torch.pow(torch.clamp(x_ij / self.x_max, max=1.0), self.alpha)
        
        # 2. Compute the squared error: (w_i^T * w_j + b_i + b_j - log(x_ij))^2
        # We use log(x_ij) as the target
        log_x_ij = torch.log(x_ij)
        squared_loss = (predictions - log_x_ij) ** 2
        
        # 3. Apply weights and sum (standard GloVe sums the total loss)
        weighted_loss = weights * squared_loss
        
        return torch.sum(weighted_loss)
    
def export_to_vec(vocab, embeddings, filename="outputs/embeddings.vec"):
    with open(filename, 'w', encoding='utf-8') as f:
        # First line: Vocab Size and Vector Dimension
        f.write(f"{len(vocab.word2id)} {embeddings.shape[1]}\n")
        
        for word, idx in vocab.word2id.items():
            vector = embeddings[idx].cpu().numpy()
            vector_str = " ".join(map(str, vector))
            f.write(f"{word} {vector_str}\n")
    print(f"Exported to standard text format: {filename}")

def save_sparse_matrix(matrix, filename="cooccurrence_matrix.pt"):
    """
    Saves the torch.sparse_coo_tensor to a file.
    """
    # Ensure the matrix is coalesced (merges duplicate indices) before saving
    if not matrix.is_coalesced():
        matrix = matrix.coalesce()
        
    torch.save(matrix, filename)
    print(f"Sparse co-occurrence matrix saved to {filename}")

if __name__=='__main__':
    vocab = Vocab()
    vocab.build_from_large_json('data/data.json')

    # N = len(vocab) #Size of Co-occurence matrix
    # X = torch.zeros()
    # X = build_cooccurrence_matrix('data/corpus.txt', vocab)
    # save_sparse_matrix(X)

    # print(X.shape)
    
    X = torch.load("cooccurrence_matrix.pt")
    print("Co-occurence Matrix Loaded")

    data = GloVeDataset(X)
    batch_size = 256
    loader = DataLoader(data, batch_size=batch_size, num_workers=4)
    d = 200
    x_max = 100
    alpha = 0.75
    num_epochs = 200
    model = GloVe(len(vocab), d)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = GloVeLoss(x_max, alpha)

    for epoch in range(num_epochs):
        total_loss = 0
        for index, (target_ids, context_ids, counts) in enumerate(loader, 1):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(target_ids, context_ids)
            
            # Calculate weighted loss
            loss = loss_fn(outputs, counts)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()/batch_size

        print('epoch: ', epoch, 'Loss: ', total_loss)

    
    export_to_vec(vocab, model.w_embeddings.weight.data + model.c_embeddings.weight.data)
    
