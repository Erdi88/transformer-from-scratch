import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(
            embed_size, num_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size)
        )

    def forward(self, x, attn_mask, padding_mask):
        x_norm = self.norm1(x)
        attn, _ = self.attention(x_norm, x_norm, x_norm, 
                                 key_padding_mask=padding_mask, 
                                 attn_mask=attn_mask, 
                                 need_weights=False, 
                                 is_causal=True)
        x_norm = self.dropout(attn)
        x = x + x_norm
        x_norm = self.norm2(x)
        x_norm = self.mlp(x_norm)
        x_norm = self.dropout(x_norm)
        x = x + x_norm
        return x



class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe_matrix = torch.zeros(max_len, embed_size)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = 10000**(torch.arange(0, embed_size, 2) / embed_size)
        #From section 3.5
        pe_matrix[:, 0::2] =torch.sin(pos/div)
        pe_matrix[:, 1::2] =torch.cos(pos/div)
        self.register_buffer("positional_encoding", pe_matrix)
    def forward(self, x):
        seq_len = x.shape[1]
        #move the positional encoding to the device of the input tensor
        pe = self.positional_encoding[:seq_len, :].to(x.device)
        return x + pe



class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        matrix = torch.ones(seq_len, seq_len)
        up_triangle = torch.triu(matrix, diagonal=1)
        mask = up_triangle.bool()
        return mask
        

if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

