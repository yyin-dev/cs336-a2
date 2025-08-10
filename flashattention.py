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
            Q: ... m d
            K: ... n d
            V: ... n d

        """
        d = Q.shape[-1]
        m = Q.shape[-2]
        n = K.shape[-2]

        Q = rearrange(Q, "... d -> (...) d")
        K = rearrange(K, "... d -> (...) d")
        V = rearrange(V, "... d -> (...) d")
        Nq = Q.shape[0]
        Nk = K.shape[0]

        O = torch.zeros((Nq, d), device=Q.device)
        L = torch.zeros((Nq,), device=Q.device)

        Bq = 8
        Bk = 16
        assert Nq % Bq == 0
        assert Nk % Bk == 0

        Tq = cdiv(Nq, Bq)
        Tk = cdiv(Nk, Bk)

        for i in range(Tq):
            Qi = Q[i * Bq : (i + 1) * Bq, :]
            Oij = O[i * Bq : (i + 1) * Bq, :]
            Lij = L[i * Bq : (i + 1) * Bq]
            mij = torch.full((Bq,), -torch.inf)
            for j in range(Tk):
                Kj = K[j * Bk : (j + 1) * Bk, :]
                Vj = V[j * Bk : (j + 1) * Bk, :]

                Sij = einsum(Qi, Kj, "Bq d, Bk d -> Bq Bk") / math.sqrt(d)
                mij_old = mij.clone().detach()  # Create a copy
                Sij_rowmax = torch.max(Sij, dim=-1).values  # (Bq,)
                mij = torch.max(mij_old, Sij_rowmax)  # (Bq,)

                # Sij is (Bq, Bk), and mij is (Bq,).
                # PyTorch's broadcasting starts with trailing dimension, in
                # this case Bk vs. Bq, and wouldn't work here.
                Pij = torch.exp(Sij - rearrange(mij, "(Bq x) -> Bq x", x=1))  # (Bq, Bk)
                Lij = torch.exp(mij_old - mij) * Lij + torch.sum(Pij, dim=-1)  # (Bq,)
                Oij = einsum(
                    torch.diag(torch.exp(mij_old - mij)), Oij, "Bq Bq, Bq d -> Bq d"
                ) + einsum(Pij, Vj, "Bq Bk, Bk d -> Bq d")

            Oij = einsum(torch.diag(1 / Lij), Oij, "Bq Bq, Bq d -> Bq d")
            Lij = torch.log(Lij) + mij

            O[i * Bq : (i + 1) * Bq, :] = Oij
            L[i * Bq : (i + 1) * Bq] = Lij

        # revert shape
        Q = rearrange(Q, "(b m) d -> b m d", m=m)
        K = rearrange(K, "(b n) d -> b n d", n=n)
        V = rearrange(V, "(b n) d -> b n d", n=n)
        L = rearrange(L, "(b m) -> b m", m=m)
        O = rearrange(O, "(b m) d -> b m d", m=m)
        print(f"O sum: {O.sum()}, L sum: {L.sum()}")

        ctx.save_for_backward(Q, K, V, L, O)

        return O

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError
