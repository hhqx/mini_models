import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape to (batch_size, seq_len, num_heads, head_dim)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, v)
        
        # Reshape back to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MiniLLM(nn.Module):
    def __init__(self, 
                 vocab_size=10000, 
                 max_seq_len=128,
                 d_model=128,     # Small hidden dimension
                 num_heads=4,     # Small number of heads
                 d_ff=256,        # Small feed-forward dimension
                 num_layers=2,    # Just a couple of layers
                 dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute position ids tensor for traceability
        self.register_buffer(
            "position_ids", 
            torch.arange(max_seq_len).expand((1, -1))
        )
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        # Use pre-computed position ids and slice to the actual sequence length
        position_ids = self.position_ids[:, :seq_length]
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.dropout(token_embeds + position_embeds)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
            
        # Output projection
        logits = self.output_layer(x)
        
        return logits


def create_mini_model(vocab_size=10000, max_seq_len=128):
    """
    Factory function to create a small model instance
    """
    return MiniLLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,    # Small hidden dimension
        num_heads=4,    # Small number of heads
        d_ff=256,       # Small feed-forward dimension
        num_layers=2,   # Just a couple of layers
        dropout=0.1
    )

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)

if __name__ == '__main__':
    model = MiniLLM(
        # vocab_size=10000,
        # max_seq_len=128,
        # d_model=128,
        # num_heads=4,
        # d_ff=256,
        # num_layers=2,
        # dropout=0.1
    )
    
    model.eval()
    
    with torch.no_grad():
        # Fix input tensor to match expected input_ids dimensions
        dummy_input = torch.randint(0, 100, (1, 10))  # batch_size=1, seq_len=10
        attention_mask = torch.ones(1, 10)  # Create appropriate attention mask
        output = model(dummy_input, attention_mask)