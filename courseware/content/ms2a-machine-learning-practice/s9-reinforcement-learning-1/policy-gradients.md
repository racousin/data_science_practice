# Policy Gradients

A value function is a means to an end: you want the policy. Policy gradient
methods skip the intermediate object and optimise the policy directly, by
gradient ascent on the expected return.

<!-- notes: 30 minutes. The log-derivative trick is the only piece of algebra
worth doing on the board — everything else follows from it. Reserve the last
eight minutes for variance: baselines and reward-to-go are what turn REINFORCE
from a curiosity into something that runs. -->

---

## Parameterise the policy


- Parametrization of policy, $\pi_\theta$.
- We aim to maximize the expected return $J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G(\tau) \right]$.
- Gradient ascent: $\theta_{k+1} = \theta_k + \alpha \left. \nabla_\theta J(\pi_\theta) \right|_{\theta_k}$.
- We can prove that:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G(\tau) \right]
$$

For more details, see: [Policy Gradient Algorithms](URL)

## Reinforce/VPG algorithm

Initialize policy $\pi_\theta$

1. Generate episodes $\mathcal{D} = \{\tau_i\}_{i=1,\dots,N}$ with the policy $\pi_\theta$
2. Compute gradient approximation

$$
\hat{\nabla} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t
$$

3. Update policy (apply gradient ascent) $\theta \leftarrow \theta + \alpha \hat{\nabla}$
4. Iterate
---

## The advantage function

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

How much better this action is than the policy's default behaviour in this
state. Positive means push the probability up, negative means push it down,
zero means the action was unremarkable and the gradient should be small.

Every algorithm from here on estimates $A$ and plugs it into the same
expression. The general form is

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, \Phi_t \right]
$$

with $\Phi_t$ one of: the full return $G_t$ (REINFORCE), the reward-to-go, the
reward-to-go minus a baseline, or an estimated advantage (actor-critic).
