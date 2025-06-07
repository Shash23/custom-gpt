import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

import os
if not os.path.exists('input.txt'):
    print("input.txt not found. Please download it from:")
    print("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    # Create a small sample text for demonstration
    text = """Hello world! This is a sample text for testing the transformer model. 
    It contains various characters and should work fine for basic testing purposes.
    The quick brown fox jumps over the lazy dog. 1234567890!@#$%^&*()"""
else:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequencies and their complex exponentials for RoPE.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes frequency tensor for broadcasting with input tensor.
    freqs_cis: Expected shape (T, head_size // 2)
    x: Expected shape (B, n_head, T, head_size // 2) (complex view of xq/xk)
    Desired output shape for freqs_cis: (1, 1, T, head_size // 2)
    """
    # Simply add two singleton dimensions at the beginning for batch and head dimensions.
    return freqs_cis.view(1, 1, *freqs_cis.shape)


def apply_rotary_pos_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies rotary positional embeddings to query and key tensors.
    The input tensors xq and xk are expected to have shape (B, n_head, T, head_size).
    """
    # xq, xk are (B, n_head, T, head_size)
    # Reshape for complex view: (B, n_head, T, head_size // 2, 2) -> (B, n_head, T, head_size // 2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis is (T, head_size // 2)
    # Reshape freqs_cis to (1, 1, T, head_size // 2) for broadcasting across B and n_head
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # xq_ is (B, n_head, T, head_size // 2) complex

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # Flatten the last two dimensions (complex and real parts)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

        # linear layer for q, k, v

        self.key = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.query = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.value = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.proj = nn.Linear(n_head * head_size, n_embd)

        # causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        #RoPE: precompute frequencies once for all heads
        self.freqs_cis = precompute_freqs_cis(head_size, block_size).to(device)


    def forward(self, x):
        

        # (B, T, n_embd) → (B, T, n_head, head_size)

        B,T,C = x.shape


        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # apply RoPE to query and key
        q_rot, k_rot = apply_rotary_pos_emb(q, k, self.freqs_cis[:T])


        wei = q_rot @ k_rot.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, n_head, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v  # (B, n_head, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)  # (B, T, n_head * head_size)
        out = self.dropout(self.proj(out))  # Final linear projection

        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# create model
model = TransformerLanguageModel()
m = model.to(device)

# print the number of parameters in the model
#print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Early Stopping variables
best_val_loss = float('inf')
patience = 10 # Number of eval_intervals to wait for validation loss improvement
patience_counter = 0

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        current_val_loss = losses['val']
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {current_val_loss:.4f}")

        # Early stopping logic
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Optional: Save the model's state_dict here if you want to load the best performing model later
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at step {iter}: Validation loss has not improved for {patience} intervals.")
                break # Exit the training loop

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))