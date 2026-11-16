#!/usr/bin/env python
"""
research/phase19_grad.py -- newly written Kronecker gradient sketch for
Phase 19 Block D, following the description in PHASE17_PREREG.md lines
286-293 (Phase 17 D1 never ran; no code is reused). Validated by Block D's
>= 0.95 sketch check.

For every LoRA-targeted linear module (36 blocks x 7 = 252), seeded Gaussian
R_in (d_in x 8) and R_out (d_out x 8) with N(0, 1/8) entries. For one scalar
objective f (a sequence's summed log-prob), the full-weight gradient of the
module's effective weight is G = sum_t delta_t x_t^T (delta = df/d output,
x = module input). The sketch is S = R_out^T G R_in = sum_t (R_out^T delta_t)
(R_in^T x_t)^T, an 8x8 block per module (64 dims x 252 modules = 16,128).
E<S, S'> = <G, G'>_F. The full gradient is never materialised.

Mechanics: a forward pre-hook stores R_in^T x for the module; a forward hook
registers a tensor hook on the module output that, during backward, forms
R_out^T delta and accumulates the 8x8 outer-product sum per batch row. Inputs
get requires_grad through inputs_embeds; no parameter requires grad.
Gradient checkpointing (non-reentrant) bounds activation memory for long
sequences; the forward pre-hook re-fires on recomputation with identical
values, and tensor hooks fire once (on the original graph's outputs).
Dropout modules are put in eval mode so the computation is deterministic.
"""
from __future__ import annotations

import contextlib

import torch

K_SIDE = 8


class Sketcher:
    def __init__(self, mm, seed: int, exact_modules: tuple = ()):
        self.mm = mm
        self.mods = [(li, qn, m) for li, qn, m in mm.lora]
        self.R_in, self.R_out = [], []
        dev = mm.device
        for idx, (_li, _qn, m) in enumerate(self.mods):
            base = m.get_base_layer()
            g = torch.Generator(device="cpu").manual_seed(seed * 1000 + idx)
            self.R_in.append((torch.randn(base.in_features, K_SIDE, generator=g) / K_SIDE ** 0.5).to(dev))
            self.R_out.append((torch.randn(base.out_features, K_SIDE, generator=g) / K_SIDE ** 0.5).to(dev))
        self.exact_modules = set(exact_modules)  # indices for which exact G is formed (sketch check)
        self.handles = []
        self.xp = {}
        self.xfull = {}
        self.S = None
        self.G = {}

    def _pre(self, idx):
        def f(module, args):
            x = args[0].detach()
            with torch.no_grad():
                self.xp[idx] = x.float() @ self.R_in[idx]
            if idx in self.exact_modules:
                self.xfull[idx] = x
        return f

    def _post(self, idx):
        def f(module, args, output):
            if not output.requires_grad:
                return
            def th(g):
                gp = g.float() @ self.R_out[idx]          # [B, T, 8]
                xp = self.xp[idx]                          # [B, T, 8]
                self.S[:, idx] += torch.einsum("bto,bti->boi", gp, xp)
                if idx in self.exact_modules:
                    gx = torch.einsum("bto,bti->boi", g.float(), self.xfull[idx].float())
                    self.G[idx] = gx if idx not in self.G else self.G[idx] + gx
            output.register_hook(th)
        return f

    @contextlib.contextmanager
    def active(self):
        for idx, (_li, _qn, m) in enumerate(self.mods):
            self.handles.append(m.register_forward_pre_hook(self._pre(idx)))
            self.handles.append(m.register_forward_hook(self._post(idx)))
        try:
            yield
        finally:
            for h in self.handles:
                h.remove()
            self.handles = []
            self.xp, self.xfull = {}, {}

    def flat(self):
        return self.S.reshape(self.S.shape[0], -1)


def prepare_for_backward(mm, checkpointing: bool = True):
    model = mm.model
    if checkpointing:
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
    else:
        model.eval()


def seq_logprob_sketch(mm, sk: Sketcher, ids_list, loss_masks, chunk: int = 1024):
    """ids_list: list of token-id lists (one batch, right-padded here).
    loss_masks: per sequence, bool list aligned to ids marking tokens whose
    log-prob enters f (token j predicted from position j-1). Returns sketches
    [B, 252*64] (fp32) and f values."""
    body = mm.model.base_model.model.model
    head = mm.model.base_model.model.lm_head
    emb = body.embed_tokens
    dev = mm.device
    B = len(ids_list)
    L = max(len(x) for x in ids_list)
    pad = mm.tok.pad_token_id if mm.tok.pad_token_id is not None else mm.tok.eos_token_id
    ids = torch.full((B, L), pad, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    lm = torch.zeros((B, L), dtype=torch.bool)
    for i, (x, m) in enumerate(zip(ids_list, loss_masks)):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
        lm[i, :len(x)] = torch.tensor(m)
    ids, att, lm = ids.to(dev), att.to(dev), lm.to(dev)
    lm[:, 0] = False
    sk.S = torch.zeros(B, len(sk.mods), K_SIDE, K_SIDE, device=dev, dtype=torch.float32)
    sk.G = {}
    with sk.active():
        e = emb(ids).detach().requires_grad_(True)
        h = body(inputs_embeds=e, attention_mask=att, use_cache=False).last_hidden_state
        sel = lm.nonzero(as_tuple=False)       # (b, j): token j predicted at j-1
        hp = h[sel[:, 0], sel[:, 1] - 1]
        tgt = ids[sel[:, 0], sel[:, 1]]
        lps = []
        for c0 in range(0, hp.shape[0], chunk):
            def f(hh, tt):
                return torch.log_softmax(head(hh).float(), -1).gather(-1, tt.unsqueeze(-1)).squeeze(-1)
            lps.append(torch.utils.checkpoint.checkpoint(f, hp[c0:c0 + chunk], tgt[c0:c0 + chunk], use_reentrant=False))
        lp = torch.cat(lps)
        per = torch.zeros(B, device=dev, dtype=torch.float32).index_add(0, sel[:, 0], lp)
        per.sum().backward()
    vals = per.detach().tolist()
    S = sk.flat().detach().clone()
    G = {k: v.detach().cpu() for k, v in sk.G.items()}
    sk.S, sk.G = None, {}
    return S, vals, G
