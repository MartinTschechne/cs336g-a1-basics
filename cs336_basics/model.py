"""Language Model related code."""

from typing import Optional
from jaxtyping import Float, Int

import numpy as np
import torch
from torch import Tensor

from cs336_basics import nn_utils


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int , device=None, dtype=None):
        super(Linear, self).__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(w)
        self.w = torch.nn.Parameter(data=w)

    def forward(self, x: Float[Tensor, "..."]) ->  Float[Tensor, "..."]:
        return x @ self.w.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        lookup = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(lookup)
        self.lookup = torch.nn.Parameter(data=lookup)
    
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, " ... d_model"]:
        return self.lookup[token_ids]


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    in_dtype = in_features.dtype
    return (in_features * torch.sigmoid(in_features.to(torch.float64))).to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.linear_1 = Linear(d_model,d_ff,device,dtype)
        self.linear_2 = Linear(d_ff,d_model,device,dtype)
        self.linear_3 = Linear(d_model,d_ff,device,dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.linear_2(silu(self.linear_1(x)) * self.linear_3(x))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k_inv = 1./ np.sqrt(K.shape[-1])
    qk = torch.einsum("...qd,...kd->...qk",Q,K) * d_k_inv
    if mask is not None:
        qk[...,~mask] = -torch.inf
    sdpa = torch.einsum("...qk,...kd->...qd",nn_utils.softmax(qk,-1),V.to(torch.float64))

    return sdpa.to(torch.float32)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, num_heads: int, d_model: int, theta: Optional[float] = None, max_seq_len: Optional[int] = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_model // num_heads
        self.linear_q = Linear(self.d_model, self.d_k * self.num_heads)
        self.linear_k = Linear(self.d_model, self.d_k * self.num_heads)
        self.linear_v = Linear(self.d_model, self.d_v * self.num_heads)
        self.linear_o = Linear(self.d_v * self.num_heads, self.d_model)
        self.rope = None
        if theta and max_seq_len:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Optional[Int[Tensor, " ... sequence_length"]] = None):
        seq_len = x.shape[-2]
        mask = torch.ones((seq_len, seq_len)).to(torch.bool)
        mask[torch.triu(mask,1)] = False
        K = self.linear_k(x).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2)
        Q = self.linear_q(x).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V = self.linear_v(x).view(-1,seq_len,self.num_heads,self.d_v).transpose(1,2)
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        heads = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask)
        heads = heads.transpose(1,2).contiguous().view(-1,seq_len,self.d_model)
        o = self.linear_o(heads)
        return o


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE,self).__init__()
        get_arg = lambda m, i: m * theta**(-2*i/d_k)
        get_rot_mat = lambda m, i: [[np.cos(get_arg(m,i)), -np.sin(get_arg(m,i))],
                                    [np.sin(get_arg(m,i)), np.cos(get_arg(m,i))]]
        R = np.zeros((max_seq_len,d_k,d_k))
        for m in range(max_seq_len):
            for i in range(d_k//2):
                i_ = 2*i
                R[m,i_:i_+2,i_:i_+2] = get_rot_mat(m,i)
        self.R = torch.DoubleTensor(R,device=device)
        self.register_buffer('rope_matrix', self.R, persistent=False)
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_k"],
                      token_positions: Optional[Int[Tensor, " ... sequence_length"]] = None
    )-> Float[Tensor, " ... sequence_length d_k"]:
        in_dtype = x.dtype
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])
        return torch.einsum("...sij,...bsj->...bsi",self.R[token_positions],x.to(torch.float64)).to(in_dtype)

# class RoPE(torch.nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#         super(RoPE,self).__init__()
#         get_arg = lambda m, i: m * theta**(-2*i/d_k)
#         self.cos = torch.Tensor([[list(chain.from_iterable(
#             [[np.cos(get_arg(m,i)), np.cos(get_arg(m,i))] 
#             for i in range(d_k//2)]))] for m in range(max_seq_len)]).squeeze()
        
#         self.sin = torch.Tensor([[list(chain.from_iterable(
#             [[-np.sin(get_arg(m,i)),np.sin(get_arg(m,i))] 
#             for i in range(d_k//2)]))] for m in range(max_seq_len)]).squeeze()
    
#     def forward(self, x: Float[Tensor, " ... sequence_length d_k"],
#                       token_positions: Int[Tensor, " ... sequence_length"]
#     )-> Float[Tensor, " ... sequence_length d_k"]:
#         x_ = x.clone()
#         for i in range(0,x.shape[-2],2):
#             x_[...,i:i+2,:] = torch.flip(x_[...,i:i+2,:],[1])
#         return x * self.cos[token_positions] + x_ * self.sin[token_positions]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        gain = torch.empty(d_model,device=device,dtype=dtype)
        torch.nn.init.trunc_normal_(gain)
        self.gain = torch.nn.Parameter(gain)
    
    def forward(self, x: Float[Tensor, " batch_size sequence_length d_model"]) -> Float[Tensor, " batch_size sequence_length d_model"]:
        in_dtype = x.dtype
        d_model = x.shape[-1]
        x = x.to(torch.float64)
        norm = torch.sqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        result = x/norm * self.gain
        return result.to(in_dtype)


class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads: int, d_model, d_ff: int, theta: int, max_seq_len: int):
        super(TransformerBlock, self).__init__()
        self.rms_norm_1 = RMSNorm(d_model,dtype=torch.float64)
        self.mhsa = MultiHeadSelfAttention(num_heads, d_model, theta, max_seq_len)
        self.rms_norm_2 = RMSNorm(d_model,dtype=torch.float64)
        self.swiglu = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Float[Tensor, "batch seq_len d_model"], token_positions: Optional[Int[Tensor,"... sequence_length"]] = None):
        y = x + self.mhsa(self.rms_norm_1(x), token_positions)
        return y + self.swiglu(self.rms_norm_2(y))


class TransformerLM(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super(TransformerLM, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.transformer_blocks = torch.nn.Sequential(*[TransformerBlock(
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            theta=rope_theta,
            max_seq_len=context_length
        ) for _ in range(num_layers)])
        self.rms_norm = RMSNorm(d_model,dtype=torch.float64)
        self.logits = Linear(d_model,vocab_size)
    
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"], token_positions=None):
        emb = self.embedding(x)
        attn_out = self.transformer_blocks(emb)
        attn_norm = self.rms_norm(attn_out)
        logits = self.logits(attn_norm)
        return logits