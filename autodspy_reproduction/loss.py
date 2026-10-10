"""Actual released objective and explicitly labeled clipped sensitivity objective."""
import torch


def group_advantages(rewards, mode="centered"):
    values = torch.as_tensor(rewards, dtype=torch.float32).detach()
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("A group requires at least two finite rewards")
    if mode not in ("centered", "standardized"):
        raise ValueError("Unknown advantage mode")
    if torch.equal(values, values[0].expand_as(values)):
        return torch.zeros_like(values)
    advantages = values - values.mean()
    if mode == "standardized":
        advantages = advantages / values.std(correction=0).clamp_min(1e-8)
    return advantages


def action_loss(current_log_prob, old_log_prob, advantage, entropy,
                objective="notebook", clip_epsilon=0.2, entropy_coefficient=0.0):
    old = torch.as_tensor(old_log_prob, device=current_log_prob.device).detach()
    adv = torch.as_tensor(advantage, device=current_log_prob.device).detach()
    if objective == "notebook":
        if entropy_coefficient != 0:
            raise ValueError("Notebook objective has no entropy term")
        return -current_log_prob * adv
    if objective != "clipped":
        raise ValueError("Unknown objective")
    ratio = torch.exp(current_log_prob - old)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * adv)
    return -surrogate - entropy_coefficient * entropy


def group_loss(current, old, rewards, entropies, objective="notebook",
               advantage_mode="centered", clip_epsilon=0.2, entropy_coefficient=0.0):
    """[K,T] arrays; sums actions/group exactly as the released notebook."""
    if current.ndim != 2 or current.shape != old.shape or current.shape != entropies.shape:
        raise ValueError("Log probabilities and entropies must have identical [K,T] shapes")
    advantages = group_advantages(rewards, advantage_mode).to(current.device)
    if advantages.numel() != current.shape[0]:
        raise ValueError("Reward count must match group size")
    return action_loss(current, old, advantages[:, None], entropies,
                       objective, clip_epsilon, entropy_coefficient).sum()
