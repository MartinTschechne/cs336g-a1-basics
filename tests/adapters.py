from __future__ import annotations

import os
import regex as re
import random
from itertools import chain, batched, cycle, pairwise
from collections import Counter, defaultdict
from functools import cache
import multiprocessing
from copy import copy

from typing import IO, Any, BinaryIO, Optional
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn.functional as F

EPS = 1e-6
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int , device=None, dtype=None):
        super(Linear, self).__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(w)
        self.w = torch.nn.Parameter(data=w)

    def forward(self, x: Float[Tensor, "..."]) ->  Float[Tensor, "..."]:
        return x @ self.w.T


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in,d_out)
    linear.load_state_dict({"w":weights})
    return linear(in_features)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        lookup = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(lookup)
        self.lookup = torch.nn.Parameter(data=lookup)
    
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, " ... d_model"]:
        return self.lookup[token_ids]


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.load_state_dict({"lookup":weights})
    return embedding(token_ids)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.linear_1 = Linear(d_model,d_ff,device,dtype)
        self.linear_2 = Linear(d_ff,d_model,device,dtype)
        self.linear_3 = Linear(d_model,d_ff,device,dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.linear_2(run_silu(self.linear_1(x)) * self.linear_3(x))

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.linear_1.load_state_dict({"w":w1_weight})
    swiglu.linear_2.load_state_dict({"w":w2_weight})
    swiglu.linear_3.load_state_dict({"w":w3_weight})
    return swiglu(in_features)


def run_scaled_dot_product_attention(
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
    sdpa = torch.einsum("...qk,...kd->...qd",run_softmax(qk,-1),V.to(torch.float64))

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
        heads = run_scaled_dot_product_attention(
            Q,
            K,
            V,
            mask)
        heads = heads.transpose(1,2).contiguous().view(-1,seq_len,self.d_model)
        o = self.linear_o(heads)
        return o


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mhsa = MultiHeadSelfAttention(num_heads, d_model)
    mhsa.load_state_dict({
        "linear_q.w": q_proj_weight,
        "linear_k.w": k_proj_weight,
        "linear_v.w": v_proj_weight,
        "linear_o.w": o_proj_weight,
    })
    return mhsa(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mhsa = MultiHeadSelfAttention(num_heads, d_model, theta, max_seq_len)
    mhsa.load_state_dict({
        "linear_q.w": q_proj_weight,
        "linear_k.w": k_proj_weight,
        "linear_v.w": v_proj_weight,
        "linear_o.w": o_proj_weight,
    })
    return mhsa(in_features, token_positions)

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


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)

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

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(num_heads, d_model, d_ff, theta, max_seq_len)
    transformer_block.load_state_dict({
        "rms_norm_1.gain": weights["ln1.weight"],
        "mhsa.linear_q.w": weights["attn.q_proj.weight"],
        "mhsa.linear_k.w": weights["attn.k_proj.weight"],
        "mhsa.linear_v.w": weights["attn.v_proj.weight"],
        "mhsa.linear_o.w": weights["attn.output_proj.weight"],
        "rms_norm_2.gain": weights["ln2.weight"],
        "swiglu.linear_1.w": weights["ffn.w1.weight"],
        "swiglu.linear_2.w": weights["ffn.w2.weight"],
        "swiglu.linear_3.w": weights["ffn.w3.weight"],
    })
    return transformer_block(in_features)

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


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    tlm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    state_dict = {
        "embedding.lookup": weights["token_embeddings.weight"],
        "rms_norm.gain": weights["ln_final.weight"],
        "logits.w": weights["lm_head.weight"]
        }
    for i in range(num_layers):
        state_dict.update({
            f"transformer_blocks.{i}.rms_norm_1.gain": weights[f"layers.{i}.ln1.weight"],
            f"transformer_blocks.{i}.mhsa.linear_q.w": weights[f"layers.{i}.attn.q_proj.weight"],
            f"transformer_blocks.{i}.mhsa.linear_k.w": weights[f"layers.{i}.attn.k_proj.weight"],
            f"transformer_blocks.{i}.mhsa.linear_v.w": weights[f"layers.{i}.attn.v_proj.weight"],
            f"transformer_blocks.{i}.mhsa.linear_o.w": weights[f"layers.{i}.attn.output_proj.weight"],
            f"transformer_blocks.{i}.rms_norm_2.gain": weights[f"layers.{i}.ln2.weight"],
            f"transformer_blocks.{i}.swiglu.linear_1.w": weights[f"layers.{i}.ffn.w1.weight"],
            f"transformer_blocks.{i}.swiglu.linear_2.w": weights[f"layers.{i}.ffn.w2.weight"],
            f"transformer_blocks.{i}.swiglu.linear_3.w": weights[f"layers.{i}.ffn.w3.weight"]
        })
    tlm.load_state_dict(state_dict)
    return tlm(in_indices)

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


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    norm = RMSNorm(d_model, eps)
    norm.load_state_dict({'gain':weights})
    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
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


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    input_seq, labels = [], []
    cl = context_length
    for start in random.sample(range(len(dataset)-cl),k=batch_size):
        input_seq.append(dataset[start:start+cl])
        labels.append(dataset[start+1:start+cl+1])
    return (torch.LongTensor(np.array(input_seq)).to(device), torch.LongTensor(np.array(labels)).to(device))


def logsumexp(x: Float[Tensor, "..."], dim: int = -1, keepdim: bool = False) -> Float[Tensor, "..."]:
    """
    """
    c = x.max().to(torch.float64)
    return c + torch.log(torch.sum(torch.exp(x.double()-c),dim,keepdim))

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    lse = logsumexp(in_features, keepdim=True)
    return torch.exp(in_features.to(torch.float64) - lse)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size, vocab_size = inputs.shape
    mask = F.one_hot(targets,vocab_size)
    lse = logsumexp(inputs)
    return -((inputs * mask).sum(-1) - lse).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    norm = np.sqrt(np.sum(np.fromiter(((param.grad.data**2).sum() for param in parameters if param.grad is not None),float)))
    if norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm/(norm + EPS)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter],
                       lr: float=1e-3,
                       weight_decay: float = 0.01,
                       betas: Tuple[float, float] = (0.9,0.999),
                       eps: float = 1e-8):
        super().__init__(params, {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps
        })
        if lr < 0:
            raise ValueError("Invalid learning rate {lr}")

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                m = state.get("m",torch.zeros_like(grad))
                m = beta_1 * m + (1-beta_1) * grad
                state["m"] = m

                v = state.get("v", torch.zeros_like(grad))
                v = beta_2 * v + (1-beta_2) * grad.pow(2)
                state["v"] = v

                t = state.get("t",1)
                alpha_t = lr * (np.sqrt(1-beta_2**t))/(1-beta_1**t)

                p.data -= alpha_t * m/(torch.sqrt(v)+group["eps"])
                p.data -= lr * weight_decay * p.data

                state["t"] = state.get("t",1) + 1


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+np.cos((it-warmup_iters)*np.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
            'epoch': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

import pdb

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)

def get_byte_pair_counts_global(freq_table: list[tuple[str], int]) -> dict[tuple[str],int]:
        byte_pair_counts = Counter()
        for pt_word, freq in freq_table:
            for byte_pair in pairwise(pt_word):
                byte_pair_counts[byte_pair] += freq
        return byte_pair_counts

class BPETokenizer:
    def __init__(self, vocab = None, merges = None, special_tokens: Optional[list[str]]=None):
        self.vocab = vocab
        self.token_to_int = None
        self.merges = merges
        self.special_tokens = sorted(special_tokens,key=len,reverse=True) if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> BPETokenizer:
        with open(vocab_filepath, 'r') as f:
            vocab = f.read()
        with open(merges_filepath, 'r') as f:
            merges = f.read()
        return cls(vocab, merges, special_tokens)

    @cache
    def _merge(self, token: tuple[bytes], merge_pair: tuple[bytes]) -> tuple[bytes]:
        if merge_pair[0] not in token:
            return token
        new_tokens = []
        j = 0
        while j < len(token):
            if j < len(token)-1 and merge_pair == (token[j], token[j+1]):
                new_tokens.append(token[j]+token[j+1])
                j += 2
            else:
                new_tokens.append(token[j])
                j += 1
        return tuple(new_tokens)
    
    def _token_to_int(self, pre_token_enc):
        if pre_token_enc in self.token_to_int:
            return [self.token_to_int[pre_token_enc]]
        return [self.token_to_int[bytes([pte])] for pte in pre_token_enc]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self.token_to_int = {v:k for k, v in self.vocab.items()}
        pre_tokenized_corpus = self._clean_input(text)

        corpus_enc = []
        for pt_doc in pre_tokenized_corpus:
            if not pt_doc:
                continue
            doc_enc = []
            for pre_token in pt_doc:
                pre_token_enc = pre_token.encode()
                if pre_token_enc not in self.token_to_int:
                    pre_token_enc = tuple(pre_token_enc[i:i+1] for i in range(len(pre_token_enc)))
                else:
                    pre_token_enc = tuple([pre_token_enc])
                
                if len(pre_token_enc) == 1:
                    doc_enc.append(self.token_to_int[pre_token_enc[0]])
                    continue
                for merge in self.merges:
                    pre_token_enc = self._merge(pre_token_enc, merge)
                doc_enc += [self.token_to_int[pte] for pte in pre_token_enc]
            corpus_enc += doc_enc# + [self.token_to_int[b'<|endoftext|>']]
        return corpus_enc#[:-1]
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[str]:
        for it in iterable:
            for token in  self.encode(it):
                yield token

    def decode(self, ids: list[int]) -> str:
        text = b""
        if ids and isinstance(ids[0],list):
            ids = chain.from_iterable(ids)
        for id_ in ids:
            text += self.vocab[id_]
        return text.decode(errors='replace')
    
    def _clean_input(self, file_content):
        if self.special_tokens:
            splitted = re.split('('+'|'.join([re.escape(st) for st in self.special_tokens])+')', file_content)
        else:
            splitted = [file_content]
        return [re.findall(PAT, spl) if spl not in self.special_tokens else [spl] for spl in splitted]

    def _get_byte_pair_counts(self, freq_table: dict[tuple[str],int]) -> dict[tuple[str],int]:
        byte_pair_counts = defaultdict(int)
        for pt_word, freq in freq_table.items():
            for byte_pair in pairwise(pt_word):
                byte_pair_counts[byte_pair] += freq
        return byte_pair_counts
    
    def _get_byte_pair_counts_parallel(self, freq_table: dict, num_workers: int = 4):
        items = batched(freq_table.items(),len(freq_table)//max(1,num_workers-1))
        with multiprocessing.Pool(processes=num_workers) as pool:
            res = pool.map(get_byte_pair_counts_global, items)
        byte_pair_counts = Counter()
        for r in res:
            byte_pair_counts += r
        return byte_pair_counts
    
    def _update_freq_table(self, freq_table: dict, max_pair: tuple):
        # update freq_table
        old_new_cnt = []
        for pt_word, freq in freq_table.items():
            if max_pair[0] in pairwise(pt_word):
                old = pt_word
                cnt = freq
                new = []
                j = 0
                while j < len(pt_word):
                    if max_pair[0] == pt_word[j:j+2]:
                        new.append(max_pair[0][0]+max_pair[0][1])
                        j += 2
                    else:
                        new.append(pt_word[j])
                        j += 1
                old_new_cnt.append((old, new, cnt))
        for old, new, cnt in old_new_cnt:
            del freq_table[old]
            freq_table[tuple(new)] = cnt
        return freq_table
    
    def train_tokenizer(self, input_path, vocab_size, special_tokens = []) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.i = 256
        for st in special_tokens:
            self.vocab[self.i] = st.encode('utf-8')
            self.i += 1
        self.merges = list()
        self.special_tokens = special_tokens

        with open(input_path, 'r') as f:
            file_content = f.read()
        
        pre_tokenized_corpus = self._clean_input(file_content)
        freq_table = Counter(tuple(pt_word.encode()[i:i+1] for i in range(len(pt_word.encode())))
            if pt_word not in self.special_tokens else tuple([pt_word.encode()])
            for pt_doc in pre_tokenized_corpus for pt_word in pt_doc
        )

        while self.i < vocab_size:
            byte_pair_counts = self._get_byte_pair_counts(freq_table)
            max_pair = max(byte_pair_counts.items(), key=lambda kv: (kv[1],kv[0]))
            self.merges.append((max_pair[0][0],max_pair[0][1]))
            self.vocab[self.i] = max_pair[0][0] + max_pair[0][1]
            self.i += 1
            freq_table = self._update_freq_table(freq_table, max_pair)
        return self.vocab, self.merges


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    bpe = BPETokenizer()
    return bpe.train_tokenizer(input_path,vocab_size,special_tokens)
