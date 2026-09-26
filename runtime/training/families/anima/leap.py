"""LeapAlign / FlowBP trajectory self-distillation training step (reward-model-free version).

Combines the two-step leap trajectory from the LeapAlign paper (arXiv
2604.15311v2) with the surrogate-trajectory design space from the FlowBP
paper (arXiv 2606.11075). Both originals backprop through a reward model;
here the reward model is dropped entirely, and "maximize reward" is replaced
with a self-distillation objective: the x_hat_0 obtained by integrating along
the surrogate trajectory should approach the dataset's true x0.

Key differences from the original LeapAlign_Code/fastvideo/train_leapalign_flux.py:
- The original must first run a full online rollout sampling trajectory to
  get x0 (very VRAM-hungry); here the dataset already has a true x0, so we
  just add noise at an arbitrary point in time -- **no rollout needed**,
  which fits LoRA naturally.
- The original loss = max(0, lambda - reward(x0)), with the only signal
  coming from the reward model; here loss = MSE(x_hat_0, x0), with the
  signal coming from real data.
- LeapAlign's mechanics are kept: two-step leap, latent connector, gradient
  discounting, traj-sim weighting; and FlowBP's "surrogate trajectory design
  space" is exposed as four variants.

Four variants (shared form: analytically construct trajectory points +
integrate x_hat_0 along the trajectory + MSE(x_hat_0, x0) + traj-sim tail):
- original  : two-step leap + straight-through connector + alpha discount (= current LeapAlign, K=2, 1 Jacobian)
- sparse    : K-point Euler replay, plain sum of direct terms (FlowBP-Sparse, zero connector / zero Jacobian)
- bridge    : two-step leap + Euler-reconstructed connector + alpha discount (FlowBP-Bridge, structurally exact, no ST bias)
- lagrange  : two-segment leap, each segment a three-point Lagrange/Simpson integral (FlowBP-Lagrange, lower per-segment integration error)

Self-distillation note: since the ground truth is an analytic straight-line
interpolation point with no rollout noise, the connector residual is
undercut at the root, so the bridge/lagrange gains over original are
narrower; sparse is the only structurally different variant (no connector +
K-point dense supervision), at the cost of K x forward passes + K x
activation memory.

Rectified flow convention (matches training/loop.py):
- t=0 is the data end, t=1 is the noise end
- x_t = (1-t)*x0 + t*x1, velocity v = x1 - x0
- one leap (from time a to time b, a>b): x_hat_b = x_a - (a-b)*v_theta(x_a, a)
"""

from __future__ import annotations

import torch

from training.model_loading import forward_with_optional_checkpoint


def sample_two_timesteps(
    bs: int,
    device,
    min_gap: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample two per-sample times (k, j), guaranteeing k > j with a gap >= min_gap, both in (0,1).

    k leans toward the noise end (large t), j toward the data end (small t).
    First sample two points in (0,1) and sort them into (hi, lo); when the
    gap is too small, push hi toward the noise end and pull lo toward the
    data end, then clamp back into the open interval.
    """
    a = torch.rand(bs, device=device, dtype=dtype)
    b = torch.rand(bs, device=device, dtype=dtype)
    k = torch.maximum(a, b)
    j = torch.minimum(a, b)

    # When the gap is below min_gap, widen it: push each end out by half the shortfall
    deficit = (min_gap - (k - j)).clamp(min=0.0) * 0.5
    k = k + deficit
    j = j - deficit

    eps = 1e-3
    k = k.clamp(min=eps + min_gap, max=1.0 - eps)
    # j's upper bound is per-sample (k - eps); clamp doesn't accept a tensor bound, so use minimum + a scalar lower bound
    j = torch.minimum(j, k - eps).clamp(min=eps)
    return k, j


def leap_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_k: torch.Tensor,
    t_j: torch.Tensor,
    *,
    nested_grad_coe: float = 0.3,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Two-step leap self-distillation; returns per-sample loss (B,) (no reduction, no external sample weighting applied).

    Args:
        model        -- Anima transformer (accepts a (B,) per-sample timestep)
        x0           -- true latent, shape (B,C,T,H,W)
        noise        -- noise x1, same shape as x0 (produced by make_noise)
        cross        -- text conditioning embedding
        pad_mask     -- padding mask
        t_k, t_j     -- per-sample times (B,), t_k > t_j
        nested_grad_coe   -- gradient discount alpha (paper Eq 9): scales the nested gradient, 0=cut off/1=no discount
        traj_sim_weighting -- whether to enable trajectory-similarity weighting (paper Eq 12)
        traj_sim_min       -- similarity-weighting lower bound tau (prevents near-identical pairs from being over-amplified)
        use_checkpoint     -- whether the model forward pass uses gradient checkpointing
    """
    # broadcast to latent dims (B,1,1,1,1)
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))

    # real noised latents (no rollout needed)
    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise

    # -- first leap (with gradient): x_k --v_k--> x_hat_{j|k} --
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_j = x_k - (k - j) * v_k

    # -- latent connector (paper Eq 6): forward value = ground truth, gradient flows back to v_k --
    x_j = x_hat_j + (x_j_real - x_hat_j).detach()

    # -- gradient discount (paper Eq 9): scale the nested gradient of the second leap w.r.t. x_j by alpha --
    if nested_grad_coe <= 0.0:
        x_j_in = x_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_j
    else:
        x_j_in = nested_grad_coe * x_j + (1.0 - nested_grad_coe) * x_j.detach()

    # -- second leap (with gradient): x_j --v_j--> x_hat_{0|j} --
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_0 = x_j - j * v_j

    # self-distillation loss + trajectory-similarity weighting (shared tail)
    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def _finalize_loss(
    x_hat_0: torch.Tensor,
    x0: torch.Tensor,
    *,
    x_hat_inter: torch.Tensor | None = None,
    x_inter_real: torch.Tensor | None = None,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
) -> torch.Tensor:
    """Loss tail shared by all four variants: self-distillation MSE(x_hat_0, x0) + optional trajectory-similarity weighting (paper Eq 12).

    Args:
        x_hat_0      -- x0 estimate integrated along the surrogate trajectory, shape (B,C,...)
        x0           -- true latent
        x_hat_inter  -- intermediate endpoint prediction (used for the d_inter term in traj-sim); None means only d_0 is used
        x_inter_real -- intermediate endpoint ground truth (paired with x_hat_inter)
        traj_sim_weighting -- whether to enable trajectory-similarity weighting
        traj_sim_min       -- similarity-weighting lower bound tau (prevents near-identical pairs from being over-amplified)

    The closer a leap lands to the real path (the smaller the residual), the
    higher its weight, which keeps wildly wrong predictions from large-span
    leaps from dominating the loss. sparse has no intermediate endpoint
    (x_hat_inter=None), so it degrades to weighting by the endpoint residual only.

    Dimensional note: original/bridge/lagrange have w_sim = 1/(d_inter + d_0)
    ~= 1/(2d), while sparse has w_sim = 1/d_0 ~= 1/d. At the same residual,
    sparse gets amplified by roughly 2x. Adaptive optimizers can absorb this,
    but when comparing original vs sparse with traj_sim both enabled, the
    effective loss magnitudes aren't directly comparable -- keep this in mind
    when comparing curves.
    """
    loss_per_sample = (x_hat_0.float() - x0.float()).pow(2).mean(
        dim=tuple(range(1, x0.ndim))
    )

    if traj_sim_weighting:
        with torch.no_grad():
            d_0 = (x0.float() - x_hat_0.float()).abs().mean(
                dim=tuple(range(1, x0.ndim))
            ).clamp(min=traj_sim_min)
            if x_hat_inter is not None and x_inter_real is not None:
                d_inter = (x_inter_real.float() - x_hat_inter.float()).abs().mean(
                    dim=tuple(range(1, x0.ndim))
                ).clamp(min=traj_sim_min)
                w_sim = 1.0 / (d_inter + d_0)
            else:
                w_sim = 1.0 / d_0
        loss_per_sample = loss_per_sample * w_sim

    return loss_per_sample


def sample_activation_timesteps(
    bs: int,
    device,
    k: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample K per-sample descending times t_1 > t_2 > ... > t_K, all in (0,1).

    Used for the FlowBP-Sparse activation set: starting from the noise end
    t_1, Euler-replay along K support points down to the data end. Uses
    "stratified jitter": split (0,1) into K equal bins and sample one
    uniform point per bin, which naturally guarantees (i) each point
    independently falls in (i/k, (i+1)/k), covering the whole trajectory;
    (ii) strictly descending order; (iii) adjacent gaps >= ~0 (the actual
    lower bound is ~0, far better than the collapse risk of the old greedy
    push-down approach).

    Returns a tensor of shape (B, K), each row in descending order.
    """
    if k < 2:
        raise ValueError(f"sample_activation_timesteps needs k>=2, got k={k}")

    # Stratified jitter: sample a uniform point within bin i, (i/k, (i+1)/k)
    # idx = 0..K-1 goes from data end -> noise end; flip to descending order (noise end first) after sampling
    idx = torch.arange(k, device=device, dtype=dtype)
    jitter = torch.rand(bs, k, device=device, dtype=dtype)
    pts = (idx + jitter) / k  # uniform within each bin, shape (B, K), ascending
    pts, _ = torch.sort(pts, dim=1, descending=True)  # descending: t_1 > ... > t_K

    # Clip to the open interval (0,1): stratified jitter naturally falls within (0,1), this is just a numeric-edge safety net
    eps = 1e-3
    pts = pts.clamp(min=eps, max=1.0 - eps)
    return pts


def sparse_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_steps: torch.Tensor,
    *,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """FlowBP-Sparse self-distillation: K-point Euler replay, plain sum of direct terms (zero connector / zero Jacobian).

    All trajectory points use analytic straight-line interpolation
    x_{t_i}=(1-t_i)x0+t_i*noise (naturally detached); gradients flow back to
    theta only through the velocities v_theta(x_{t_i}, t_i). Euler
    telescoping sum:

        x_hat_0 = x_{t_1} - sum_{i=1..K} (t_i - t_{i+1}) * v_theta(x_{t_i}, t_i),    t_{K+1}:=0

    When v_theta == the true velocity (noise-x0), each term
    (t_i-t_{i+1})*v = x_{t_i}-x_{t_{i+1}}, and the series telescopes exactly
    back to x_{t_1}-(x_{t_1}-x0)=x0. The supervision is a consistency check
    on the weighted integral of the velocity at K times -- a multi-point
    dense generalization of plain flow-matching's single-point MSE.

    Args:
        t_steps -- per-sample descending times (B, K), produced by sample_activation_timesteps
    """
    bs, k = t_steps.shape
    view = (-1, *([1] * (x0.ndim - 1)))

    # starting point x_{t_1} (analytic, detached)
    t1 = t_steps[:, 0].reshape(*view)
    x_hat_0 = (1.0 - t1) * x0 + t1 * noise

    # for each support point: analytically build the noised point (detached), forward to get velocity (with gradient), subtract the Euler step
    for i in range(k):
        t_i = t_steps[:, i]
        t_i_b = t_i.reshape(*view)
        x_i = (1.0 - t_i_b) * x0 + t_i_b * noise  # analytic point, no gradient to theta
        v_i = forward_with_optional_checkpoint(
            model, x_i, t_i.reshape(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
        )
        # step size h_i = t_i - t_{i+1}; last segment uses t_{K+1}:=0 (h_i = t_i)
        if i + 1 < k:
            h_i = (t_i - t_steps[:, i + 1]).reshape(*view)
        else:
            h_i = t_i_b
        x_hat_0 = x_hat_0 - h_i * v_i

    # sparse has no intermediate endpoint, so traj-sim weights by the endpoint residual only
    return _finalize_loss(
        x_hat_0, x0,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def bridge_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_k: torch.Tensor,
    t_j: torch.Tensor,
    *,
    nested_grad_coe: float = 0.3,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """FlowBP-Bridge self-distillation: two-step leap + Euler-reconstructed connector (no straight-through bias).

    The only difference from original is the connector: original swaps x_j's
    forward value for the ground truth x_j_real (straight-through: forward
    value = ground truth, gradient flows back to v_k); bridge uses the
    Euler-reconstructed value x_hat_j directly as the second leap's input,
    which is structurally exact and unbiased (the forward value is its own
    trajectory point), with the gradient propagating across the segment
    through a single alpha-scaled Jacobian d(x_hat_j)/d(v_k). alpha=0
    degrades to training only the second leap; alpha=1 is the full single
    Jacobian.
    """
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))

    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise

    # first leap (with gradient)
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_j = x_k - (k - j) * v_k

    # Euler-reconstructed connector: use x_hat_j directly (don't swap in the ground truth), discount the gradient by alpha
    if nested_grad_coe <= 0.0:
        x_j_in = x_hat_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_hat_j
    else:
        x_j_in = nested_grad_coe * x_hat_j + (1.0 - nested_grad_coe) * x_hat_j.detach()

    # second leap (with gradient): starts from the reconstructed endpoint
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    x_hat_0 = x_j_in - j * v_j

    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )


def lagrange_training_step(
    model,
    x0: torch.Tensor,
    noise: torch.Tensor,
    cross: torch.Tensor,
    pad_mask: torch.Tensor,
    t_k: torch.Tensor,
    t_j: torch.Tensor,
    *,
    nested_grad_coe: float = 0.3,
    traj_sim_weighting: bool = False,
    traj_sim_min: float = 0.1,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """FlowBP-Lagrange self-distillation: keeps the two-segment leap topology, but each segment integrates via a three-point Lagrange rule (paper SA.2).

    Three-support-point Lagrange-interpolation integration, which degrades
    to Simpson's rule for equally spaced points, with weights 1/6*[1, 4, 1]:
    the paper (SA.2) calls this "Simpson-like positive weights". This is two
    orders more accurate than original/bridge's single-point Euler (error
    O(dt^2)) -- Simpson's error is O(dt^5).

    Per-segment integral form (segment [b,a], a>b, leaping from a to b, three
    points a > m > b, m=(a+b)/2):

        integral_b^a v dt ~= (a-b)/6 * [v(x_a) + 4*v(x_m) + v(x_b)]

    All three points need a velocity: endpoint x_a, midpoint x_m, and the
    other endpoint x_b. **Gradient constraint** (paper Eq 4/24, described as
    "at most one Jacobian factor"): every active support point other than
    the bridge anchor gets a stop-gradient on its latent input --
    v_theta(sg(x_i), sigma_i), so the gradient flows through theta only, not
    back to x_i. This self-distillation implementation has no rollout, so
    the cached latent at every time is just the analytic straight-line
    interpolation ground truth, so all three points use the analytic ground
    truth as input (naturally detached, no dependence on theta):
      - endpoint x_a, midpoint x_m=(1-m)x0+m*noise;
      - the other endpoint x_b's ground truth is **known** under
        self-distillation: for the first segment, the segment endpoint b=j
        is exactly x_j_real; for the second segment, the segment endpoint
        b=0 is exactly x0 itself (evaluated at t=0).
    This matches the paper's use of the cached-latent velocity, and avoids
    the undiscounted nested Jacobian (d v(x_hat_b) / d v(x_a)) that the old
    version introduced by plugging in the Euler-predicted point
    x_hat_b=x_a-(a-b)*v(x_a) -- the latter would bypass alpha and violate the
    single-Jacobian constraint. The only place a latent Jacobian is kept is
    the second segment's starting point x_j (the bridge anchor, via the
    alpha-scaled connector).

    Cost: 3 forward passes per segment (endpoint + midpoint + other
    endpoint), 6 forward passes total across both segments -- roughly 6x the
    compute + memory, the heaviest of the four variants. The two segments
    are chained: first segment k->j, connector (straight-through, same as
    original) + alpha discount, then second segment j->0.
    """
    k = t_k.view(-1, *([1] * (x0.ndim - 1)))
    j = t_j.view(-1, *([1] * (x0.ndim - 1)))
    m1 = (k + j) * 0.5  # midpoint of the first segment
    t_m1 = (t_k + t_j) * 0.5

    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise
    x_m1_real = (1.0 - m1) * x0 + m1 * noise  # analytic midpoint ground truth (detached)

    # -- first segment: three-point Lagrange / Simpson integral over x_k, x_m1, x_j_real --
    v_k = forward_with_optional_checkpoint(
        model, x_k, t_k.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_m1 = forward_with_optional_checkpoint(
        model, x_m1_real, t_m1.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    # velocity at the segment endpoint (j): under self-distillation the ground truth is already known = x_j_real (analytic, detached), not an Euler-predicted point --
    # this avoids the undiscounted nested Jacobian (paper Eq 4: non-anchor support points use v_theta(sg(x_i)))
    v_j_seg1 = forward_with_optional_checkpoint(
        model, x_j_real, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_seg1 = (v_k + 4.0 * v_m1 + v_j_seg1) / 6.0  # Simpson weights 1:4:1
    x_hat_j = x_k - (k - j) * v_seg1

    # connector (straight-through, same as original) + alpha discount
    x_j = x_hat_j + (x_j_real - x_hat_j).detach()
    if nested_grad_coe <= 0.0:
        x_j_in = x_j.detach()
    elif nested_grad_coe >= 1.0:
        x_j_in = x_j
    else:
        x_j_in = nested_grad_coe * x_j + (1.0 - nested_grad_coe) * x_j.detach()

    # -- second segment: three-point Lagrange / Simpson integral over x_j, x_m2, x0 --
    m2 = j * 0.5  # midpoint of the second segment (midpoint of j->0 is j/2)
    t_m2 = t_j * 0.5
    x_m2_real = (1.0 - m2) * x0 + m2 * noise
    v_j = forward_with_optional_checkpoint(
        model, x_j_in, t_j.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_m2 = forward_with_optional_checkpoint(
        model, x_m2_real, t_m2.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    # velocity at the segment endpoint (0): under self-distillation the ground truth is already known = x0 (evaluated at t=0), not an Euler-predicted point
    t_0 = torch.zeros_like(t_j)
    v_0_seg2 = forward_with_optional_checkpoint(
        model, x0, t_0.view(-1, 1), cross, pad_mask, use_checkpoint=use_checkpoint,
    )
    v_seg2 = (v_j + 4.0 * v_m2 + v_0_seg2) / 6.0  # Simpson weights 1:4:1
    x_hat_0 = x_j_in - j * v_seg2

    return _finalize_loss(
        x_hat_0, x0,
        x_hat_inter=x_hat_j, x_inter_real=x_j_real,
        traj_sim_weighting=traj_sim_weighting, traj_sim_min=traj_sim_min,
    )
