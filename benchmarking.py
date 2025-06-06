import torch
import time
import torch.nn as nn
import torch.nn.functional as F

# Config
B, T, C = 16, 32, 64
n_head = 4
head_size = C // n_head
n_embd = C
dropout = 0.0
block_size = T
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Input
x = torch.randn(B, T, C).to(device)

# Old style: with Head class (redefine here if needed)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class OldMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



# Instantiate models
old_mha = OldMultiHeadAttention(n_head, head_size).to(device)
new_mha = MultiHeadAttention(n_head, head_size).to(device)

# Warm-up
_ = old_mha(x)
_ = new_mha(x)

# Timing function
def time_model(model, x, steps=100):

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(steps):
        _ = model(x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    
    return (time.time() - start) / steps


print("Benchmarking 100 steps...")
old_time = time_model(old_mha, x)
new_time = time_model(new_mha, x)

print(f"Old MHA (looped heads):  {old_time * 1000:.3f} ms per step")
print(f"New MHA (batched heads): {new_time * 1000:.3f} ms per step")
print(f"Speedup: {old_time / new_time:.2f}x faster")
