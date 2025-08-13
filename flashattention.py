import torch
import math
from einops import einsum, rearrange
import os

# CPU interpreter for debugging. Needs to be set *before* triton is imported.
# os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl


def cdiv(a, b):
    return (a + b - 1) // b


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def flashattention_bwd(Q, K, V, O, dO, L, is_causal):
    """
    Args
        Q:  b m d
        K:  b n d
        V:  b n d
        O:  b m d
        dO: b m d
        L:  b m
    """
    m = Q.shape[-2]
    n = K.shape[-2]
    d = Q.shape[-1]
    S = einsum(Q, K, "b m d, b n d -> b m n") / math.sqrt(d)
    P = torch.exp(S - rearrange(L, "b (m x) -> b m x", x=1))  # b m n

    if is_causal:
        # In the forward pass, the attention score S = mask(QK/sqrt(d)).
        # The masking sets Sij to -inf for i < j.
        # The probabilities P = softmax(S). The masked entries are essentially
        # -inf, their softmax probabilities becomes essentially 0 - not
        # attending to future tokens. In other words, the masked logits never
        # affect the output.
        #
        # In the backward pass, everything words the same as the non-causal
        # case except that there's no gradient flowing back through Pij's that
        # are zeros for dS = P * (dP - D).
        #
        # For implementation, just need to ensure that the mask is applied
        # consistently: by masking either P or dS.
        q_indices = rearrange(torch.arange(0, m, device=Q.device), "(m x) -> m x", x=1)
        k_indices = rearrange(torch.arange(0, n, device=Q.device), "(n x) -> x n", x=1)
        mask = q_indices < k_indices
        P = torch.masked_fill(P, mask, 0.0)

    dV = einsum(P, dO, "b m n, b m d -> b n d")
    dP = einsum(dO, V, "b m d, b n d -> b m n")
    D = torch.sum(O * dO, dim=-1, keepdim=True)  # b m 1
    dS = P * (dP - D)
    dQ = einsum(dS, K, "b m n, b n d -> b m d") / math.sqrt(d)
    dK = einsum(dS, Q, "b m n, b m d -> b n d") / math.sqrt(d)
    return dQ, dK, dV


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args
            Q: b m d
            K: b n d
            V: b n d

        """
        B = Q.shape[0]
        Nq = Q.shape[1]
        Nk = K.shape[1]
        d = Q.shape[2]

        Bq = 8
        Bk = 16
        assert Nq % Bq == 0
        assert Nk % Bk == 0
        Tq = cdiv(Nq, Bq)
        Tk = cdiv(Nk, Bk)

        O = torch.zeros((B, Nq, d), device=Q.device)
        L = torch.zeros((B, Nq), device=Q.device)

        for b in range(B):
            for i in range(Tq):
                Qi = Q[b][i * Bq : (i + 1) * Bq, :]
                Oij = O[b][i * Bq : (i + 1) * Bq, :]
                Lij = L[b][i * Bq : (i + 1) * Bq]
                mij = torch.full((Bq,), -torch.inf, device=Q.device)
                for j in range(Tk):
                    Kj = K[b][j * Bk : (j + 1) * Bk, :]
                    Vj = V[b][j * Bk : (j + 1) * Bk, :]

                    Sij = einsum(Qi, Kj, "Bq d, Bk d -> Bq Bk") / math.sqrt(d)
                    mij_old = mij.clone().detach()  # Create a copy
                    Sij_rowmax = torch.max(Sij, dim=-1).values  # (Bq,)
                    mij = torch.max(mij_old, Sij_rowmax)  # (Bq,)

                    # Sij is (Bq, Bk), and mij is (Bq,).
                    # PyTorch's broadcasting starts with trailing dimension, in
                    # this case Bk vs. Bq, and wouldn't work here.
                    # (Bq, Bk)
                    Pij = torch.exp(Sij - rearrange(mij, "(Bq x) -> Bq x", x=1))

                    # (Bq,)
                    Lij = torch.exp(mij_old - mij) * Lij + torch.sum(Pij, dim=-1)

                    # (Bq, d)
                    Oij = einsum(
                        torch.diag(torch.exp(mij_old - mij)), Oij, "Bq Bq, Bq d -> Bq d"
                    ) + einsum(Pij, Vj, "Bq Bk, Bk d -> Bq d")

                Oij = einsum(torch.diag(1 / Lij), Oij, "Bq Bq, Bq d -> Bq d")
                Lij = torch.log(Lij) + mij

                O[b][i * Bq : (i + 1) * Bq, :] = Oij
                L[b][i * Bq : (i + 1) * Bq] = Lij

        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):  # pyright: ignore[reportIncompatibleMethodOverride]
        Q, K, V, L, O = ctx.saved_tensors
        compiled_bwd = torch.compile(flashattention_bwd)
        dQ, dK, dV = compiled_bwd(Q, K, V, O, dO, L, ctx.is_causal)
        return (dQ, dK, dV, None)  # None is needed for is_causal?


@triton.jit
def flashattention_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # When TRITON_INTERPRET=1, print() works in kernel.
    # Otherwise, needs to use tl.device_print()

    # Launch grid is (Tq, batch_size)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer with corresponding batch index, multiplied with
    # the batch stride for each tensor
    # Note the difference between `shape` and `block_shape`!
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Oij = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    Lij = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mij_old = tl.full((Q_TILE_SIZE,), -torch.inf, dtype=tl.float32)
    mij = tl.full((Q_TILE_SIZE,), -torch.inf, dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            # Compare (x, 1) with (1, y) gives (x, y)
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = tl.reshape(q_indices, (Q_TILE_SIZE, 1)) >= tl.reshape(
                k_indices, (1, K_TILE_SIZE)
            )
            Sij_discounted = Sij - 1e6
            Sij = tl.where(mask, Sij, Sij_discounted)

        Sij_rowmax = tl.max(Sij, axis=-1)

        mij = tl.maximum(mij_old, Sij_rowmax)

        Pij = tl.exp(Sij - tl.reshape(mij, (Q_TILE_SIZE, 1)))

        Lij = tl.exp(mij_old - mij) * Lij + tl.sum(Pij, axis=-1)

        # mij: (Q_TILE_SIZE,)
        # Oij: (Q_TILE_SIZE, D)
        # The diagonal matrix is implemented with element-wise multplication.
        # When A is a diagonal matrix, AB just scales rows in B by the
        # corresponding diagonal element in A.
        Oij = tl.reshape(tl.exp(mij_old - mij), (Q_TILE_SIZE, 1)) * Oij + tl.dot(
            Pij, Vj
        )

        mij_old = mij
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oij = tl.reshape(1 / Lij, (Q_TILE_SIZE, 1)) * Oij
    Lij = tl.log(Lij) + mij

    tl.store(O_block_ptr, Oij, boundary_check=(0, 1))
    tl.store(L_block_ptr, Lij, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, q_tile_size=32, k_tile_size=32):
        """
        Args
            Q: ... m d
            K: ... n d
            V: ... n d
            is_causal: whether to apply causal masking
            q_tile_size: tile size for queries (default 32)
            k_tile_size: tile size for keys (default 32)
        """

        B = Q.shape[0]
        m = Q.shape[1]
        n = K.shape[1]
        d = Q.shape[2]

        O = torch.empty((B, m, d), device=Q.device)
        L = torch.empty((B, m), device=Q.device)

        ctx.Q_TILE_SIZE = q_tile_size
        ctx.K_TILE_SIZE = k_tile_size
        ctx.is_causal = is_causal

        assert (
            Q.is_contiguous()
            and K.is_contiguous()
            and V.is_contiguous()
            and O.is_contiguous()
            and V.is_contiguous()
        )

        flashattention_fwd[(cdiv(m, ctx.Q_TILE_SIZE), B)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=m,
            N_KEYS=n,
            scale=1 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )

        ctx.save_for_backward(Q, K, V, L, O)

        return O

    @staticmethod
    def backward(ctx, dO):  # pyright: ignore[reportIncompatibleMethodOverride]
        Q, K, V, L, O = ctx.saved_tensors
        compiled_bwd = torch.compile(flashattention_bwd)
        dQ, dK, dV = compiled_bwd(Q, K, V, O, dO, L, ctx.is_causal)
        return (dQ, dK, dV, None, None, None)  # None is needed for is_causal?
