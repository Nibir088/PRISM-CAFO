import math
from typing import Tuple
import torch
import torch.nn.functional as F

# ---------- Focus / Faithfulness ----------
def soft_iou(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    inter = (pred * target).sum((1,2,3))
    union = (pred + target - pred * target).sum((1,2,3)).clamp_min(eps)
    return (inter / union).mean()

def L_align(S, M_inf): return 1.0 - soft_iou(S, M_inf)

def L_focus(S, M_inf, eps=1e-6):
    inside = (S * M_inf).sum((1,2,3))
    total  = S.sum((1,2,3)).clamp_min(eps)
    return (1.0 - inside / total).mean()

def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()
    dy = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()
    return dx + dy

def L_sparse(S: torch.Tensor) -> torch.Tensor:
    return S.mean()

# ---------- Shape priors ----------
def moment_rectangularity(M: torch.Tensor, eps=1e-6) -> torch.Tensor:
    B, _, H, W = M.shape
    y = torch.linspace(-1,1,H, device=M.device).view(1,1,H,1)
    x = torch.linspace(-1,1,W, device=M.device).view(1,1,1,W)
    mass = M.sum((1,2,3)).clamp_min(eps).view(B,1,1,1)
    mx = (M*x).sum((1,2,3), keepdim=True) / mass
    my = (M*y).sum((1,2,3), keepdim=True) / mass
    x0, y0 = x - mx, y - my
    Ixx = (M*(x0**2)).sum((1,2,3))
    Iyy = (M*(y0**2)).sum((1,2,3))
    Ixy = (M*(x0*y0)).sum((1,2,3))
    tr, det = Ixx+Iyy, Ixx*Iyy - Ixy*Ixy
    lam1 = (tr + torch.sqrt((tr**2 - 4*det).clamp_min(0))) / 2
    lam2 = (tr - torch.sqrt((tr**2 - 4*det).clamp_min(0))) / 2
    ratio = lam1.clamp_min(eps)/lam2.clamp_min(eps)  # >=1
    return torch.sigmoid(torch.log(ratio))           # (0,1)

def L_rect(M_barn: torch.Tensor) -> torch.Tensor:
    return (1.0 - moment_rectangularity(M_barn)).mean()

def soft_area(M: torch.Tensor) -> torch.Tensor:
    H, W = M.shape[-2:]
    return M.sum((1,2,3)) / (H*W)

def soft_perimeter(M: torch.Tensor, eps=1e-6) -> torch.Tensor:
    gy = M[:,:,1:,:] - M[:,:,:-1,:]
    gx = M[:,:,:,1:] - M[:,:,:,:-1]
    return (gy.abs().mean() + gx.abs().mean()).clamp_min(eps)

def soft_circularity(M: torch.Tensor) -> torch.Tensor:
    A = soft_area(M)                 # normalized area
    P = soft_perimeter(M)            # |∇M| proxy
    return (4 * math.pi * A / (P**2 + 1e-6)).clamp(0,1)

def L_circ(M_pond: torch.Tensor, tau=0.7) -> torch.Tensor:
    return F.relu(tau - soft_circularity(M_pond)).mean()
def chamfer_distance(mask_a, mask_b):
    # Input: (B, 1, H, W), binary masks
    B, _, H, W = mask_a.shape
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, H, device=mask_a.device),
        torch.linspace(0, 1, W, device=mask_a.device)
    ), dim=-1)  # [H, W, 2]

    coords = coords.view(1, 1, H, W, 2)  # [1, 1, H, W, 2]
    mask_a_pts = (mask_a.unsqueeze(-1) * coords).view(B, -1, 2)
    mask_b_pts = (mask_b.unsqueeze(-1) * coords).view(B, -1, 2)

    a_valid = (mask_a.view(B, -1) > 0).float()
    b_valid = (mask_b.view(B, -1) > 0).float()

    # B x Na x 2, B x Nb x 2
    d1 = torch.cdist(mask_a_pts, mask_b_pts).min(dim=2).values * a_valid
    d2 = torch.cdist(mask_b_pts, mask_a_pts).min(dim=2).values * b_valid
    # print(d1, d2)

    chamfer = (d1.sum(1) / (a_valid.sum(1)+1e-6) + d2.sum(1) / (b_valid.sum(1)+1e-6)) / 2
    chamfer = chamfer / (2 ** 0.5)
    return chamfer.unsqueeze(1)

# ---------- Relational prior ----------
def soft_center(M: torch.Tensor, eps=1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = M.shape
    y = torch.linspace(0,1,H, device=M.device).view(1,1,H,1)
    x = torch.linspace(0,1,W, device=M.device).view(1,1,1,W)
    mass = M.sum((2,3), keepdim=True).clamp_min(eps)
    cy = (M*y).sum((2,3), keepdim=True) / mass
    cx = (M*x).sum((2,3), keepdim=True) / mass
    return cy.squeeze(-1).squeeze(-1), cx.squeeze(-1).squeeze(-1)

def L_dist(M_barn: torch.Tensor, M_pond: torch.Tensor, tau_d=0.2, sigma=0.05) -> torch.Tensor:
    cy_b, cx_b = soft_center(M_barn)
    cy_p, cx_p = soft_center(M_pond)
    d = torch.sqrt((cy_b-cy_p)**2 + (cx_b-cx_p)**2)
    z = (d - tau_d) / sigma
    return torch.sigmoid(z).mean()

# ---------- Small helpers ----------
def union_mask(M7): 
    if M7 is None: return None
    return M7.sum(1, keepdim=True).clamp(0,1)
# Sufficiency/Necessity (counterfactuals)
def suff_nec_losses(
    f_logits,
    x: torch.Tensor, M_up: torch.Tensor, y: torch.Tensor,
    eps=0.05, m=0.10
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        base = f_logits(x).gather(1, y.view(-1,1)).squeeze(1)
    xin, xout = x * M_up, x * (1.0 - M_up)
    fin  = f_logits(xin).gather(1, y.view(-1,1)).squeeze(1)
    fout = f_logits(xout).gather(1, y.view(-1,1)).squeeze(1)
    Lsuf = torch.relu((base - fin) - eps).mean()
    Lnec = torch.relu((fout - base) + m).mean()
    return Lsuf, Lnec
