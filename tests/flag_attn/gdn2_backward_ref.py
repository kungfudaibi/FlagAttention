"""Independent PyTorch/FP64 GDN2 recurrence reference with analytic adjoint.

Semantics (per head, per token t), matching the accepted GDN2-001 reference:

    hd   = h_{t-1} * exp(g_t)
    e    = (b_t * k_t)^T @ hd
    h_t  = hd + k_t (x) (w_t * v_t - e)
    o_t  = scale * q_t^T @ h_t

``o`` uses the post-update state and the final state is ``h_T``.  Everything
runs in float64 on CPU and is independent of any device kernel.  The adjoint is
hand-written in :func:`gdn2_backward` and validated against ``torch.autograd``
and central finite differences by the accompanying test module.
"""

from __future__ import annotations

import torch

__all__ = ["gdn2_forward", "gdn2_backward", "finite_difference_check"]


def gdn2_forward(q, k, v, g, b, w, scale, h0):
    """FP64 token recurrence.

    q/k/v/g/b/w are [B,T,H,K] (or [B,T,H,V] for v/w); h0 is [B,H,K,V].
    Returns (o [B,T,H,V], ht [B,H,K,V]).
    """
    q, k, v, g, b, w = (x.double() for x in (q, k, v, g, b, w))
    h = h0.double().clone()
    outs = []
    for t in range(q.shape[1]):
        kt, bt, gt = k[:, t], b[:, t], g[:, t]
        vt, wt = v[:, t], w[:, t]
        hd = h * torch.exp(gt)[..., :, None]
        e = torch.einsum("bhk,bhkv->bhv", bt * kt, hd)
        h = hd + torch.einsum("bhk,bhv->bhkv", kt, wt * vt - e)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], h) * scale)
    return torch.stack(outs, dim=1), h


def gdn2_backward(q, k, v, g, b, w, do, dht, scale, h0):
    """Hand-written reverse-mode adjoint of :func:`gdn2_forward`.

    ``do`` is dL/do [B,T,H,V]; ``dht`` is dL/dh_T [B,H,K,V] or None.
    Returns a dict with keys q, k, v, g, b, w and h0 (dL/d initial state).
    """
    q, k, v, g, b, w = (x.double() for x in (q, k, v, g, b, w))
    do = do.double()
    B, T, H, K = q.shape
    V = v.shape[-1]

    # Recompute and keep the forward states so the adjoint stays explicit.
    h = h0.double().clone()
    states = []
    for t in range(T):
        kt, bt, gt = k[:, t], b[:, t], g[:, t]
        hd = h * torch.exp(gt)[..., :, None]
        e = torch.einsum("bhk,bhkv->bhv", bt * kt, hd)
        h = hd + torch.einsum("bhk,bhv->bhkv", kt, w[:, t] * v[:, t] - e)
        states.append((hd, h.clone()))

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g)
    db = torch.zeros_like(b)
    dw = torch.zeros_like(w)
    dh = (
        torch.zeros(B, H, K, V, dtype=torch.float64)
        if dht is None
        else dht.double().clone()
    )

    for t in range(T - 1, -1, -1):
        hd, h = states[t]
        kt, bt, gt = k[:, t], b[:, t], g[:, t]
        vt, wt = v[:, t], w[:, t]

        # o_t = scale * q_t^T h_t
        dh = dh + scale * torch.einsum("bhk,bhv->bhkv", q[:, t], do[:, t])
        dq[:, t] = scale * torch.einsum("bhkv,bhv->bhk", h, do[:, t])

        # h = hd + k (x) u,  u = w*v - e
        u = wt * vt - torch.einsum("bhk,bhkv->bhv", bt * kt, hd)
        du = torch.einsum("bhk,bhkv->bhv", kt, dh)
        de = -du

        # e = (b*k)^T hd
        dbk = torch.einsum("bhkv,bhv->bhk", hd, de)  # dL/d(b*k)
        dk[:, t] = torch.einsum("bhkv,bhv->bhk", dh, u) + dbk * bt
        db[:, t] = dbk * kt

        # hd = h_{t-1} * exp(g); dL/dhd feeds both e and the next step.
        dhd = dh + torch.einsum("bhk,bhv->bhkv", bt * kt, de)
        dg[:, t] = torch.einsum("bhkv,bhkv->bhk", dhd, hd)

        dv[:, t] = du * wt
        dw[:, t] = du * vt
        dh = dhd * torch.exp(gt)[..., :, None]

    return {"q": dq, "k": dk, "v": dv, "g": dg, "b": db, "w": dw, "h0": dh}


def finite_difference_check(q, k, v, g, b, w, scale, h0, do, dht=None, eps=1e-6):
    """Central differences of a scalar objective vs the analytic adjoint.

    Objective: sum(do * o) + sum(dht * ht). Returns {name: max_abs_diff}.
    """
    names = ("q", "k", "v", "g", "b", "w", "h0")
    tensors = {n: t.detach().double().clone() for n, t in
               zip(names, (q, k, v, g, b, w, h0))}
    if dht is None:
        dht = torch.zeros_like(tensors["h0"])

    def objective(vals):
        o, ht = gdn2_forward(
            vals["q"], vals["k"], vals["v"], vals["g"], vals["b"], vals["w"],
            scale, vals["h0"],
        )
        return (o * do.double()).sum() + (ht * dht.double()).sum()

    analytic = gdn2_backward(
        tensors["q"], tensors["k"], tensors["v"], tensors["g"],
        tensors["b"], tensors["w"], do, dht, scale, tensors["h0"],
    )
    out = {}
    for name in names:
        base = tensors[name]
        num = torch.zeros_like(base).reshape(-1)
        flat = base.reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            hi = objective(tensors)
            flat[i] = orig - eps
            lo = objective(tensors)
            flat[i] = orig
            num[i] = (hi - lo) / (2 * eps)
        out[name] = (num.reshape(base.shape) - analytic[name]).abs().max().item()
    return out
