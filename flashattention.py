import torch
import math
from einops import einsum, rearrange


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args
            Q: ... m d
            K: ... n d
            V: ... n d
        """

        d = Q.shape[-1]

        S = einsum(Q, K, "... m d, ... n d -> ... m n") / math.sqrt(d)
        L = torch.exp(S).sum(dim=-1).log()
        P = torch.softmax(S, dim=-1)
        O = einsum(P, V, "... m n, ... n d -> ... m d")

        ctx.save_for_backward(Q, K, V, L, O)

        return O

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError


def cdiv(a, b):
    return -(a // -b)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


class FlashAttentionKernelPytorch(torch.autograd.Function):
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
                mij = torch.full((Bq,), -torch.inf)
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
        return O

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError
