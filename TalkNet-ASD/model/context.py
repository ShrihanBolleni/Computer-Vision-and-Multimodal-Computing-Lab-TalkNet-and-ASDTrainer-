import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class ContextAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(ContextAttentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # apply learnable function f(target_speaker, *context_speakers)
        # at every time t

        # input shapes in format of (Num, Time, Features):
        # target speaker tensor: (1, 150, 128)
        # context speakers tensor: (3, 150, 128)
        # temporal dimension can vary, may not be 150

        # transpose speakers tensors so that temporal dimension is the batch:
        # (N, T, F) -> (T, N, F)
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)

        # softmax(Q * K.T / sqrt(d_k)) * V
        # (1 x 128) * (3 x 128).T * (3 x 128)
        # (1 x 128) * (128 x 3) * (3 x 128)
        # (1 x 3) * (3 x 128)
        # (1 x 128)
        # Repeated along temporal dimension:
        # (1 x T x 128)

        # type: (torch.Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2 = self.self_attn(query=tar, key=src, value=src, 
                              attn_mask=None,
                              key_padding_mask=None)[0]
        
        tar = tar + self.dropout1(src2)
        tar = self.norm1(tar)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(tar))))
        tar = tar + self.dropout2(src2)
        tar = self.norm2(tar)
        tar = tar.transpose(0, 1) # T, B, C -> B, T, C
        return tar
