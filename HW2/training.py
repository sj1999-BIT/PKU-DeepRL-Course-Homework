import torch
import torch.nn.functional as F


def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs_old, advantages,
               returns, clip_ratio=0.2, epochs=10):
    for _ in range(epochs):
        # Get current policy evaluation
        log_probs, entropy = policy_net.evaluate(states, actions)

        # Compute policy ratio (π_θ / π_θ_old)
        ratio = torch.exp(log_probs - log_probs_old)

        # Compute clipped objective
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

        # Add entropy bonus
        policy_loss = policy_loss - 0.01 * entropy.mean()

        # Update policy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value function
        value_pred = value_net(states)
        value_loss = F.mse_loss(value_pred, returns)

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()