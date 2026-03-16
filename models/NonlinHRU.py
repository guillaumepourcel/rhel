from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


def simple_uniform_init(rng, shape, std=1.0):
    weights = random.uniform(rng, shape) * 2.0 * std - std
    return weights


def kinetic_hamiltonian(z):
    """
    Compute the kinetic Hamiltonian of the NonlinHRU system.

    Args:
        z (float): first component of the state of the system (momentum), shape (P,)
    
    Returns:
        hamiltonian (float): scalar real value representing the kinetic energy
    """
    return 0.5 * sum(z**2)


def potential_hamiltonian(y, Bu, W_diag, b, c, alpha):
    """
    Compute the potential Hamiltonian of the NonlinHRU system.

    Args:
        y (float): second component of the state of the system (position), shape (P,)
        Bu (float): input forcing term (B @ u), shape (P,)
        W_diag (float): diagonal state matrix, shape (P,)
        b (float): bias vector, shape (P,)
        c (float): scaling parameter vector, shape (P,)
        alpha (float): potential energy parameter, shape (1,) or scalar
    
    Returns:
        hamiltonian (float): scalar real value representing the potential energy
    """
    c_reshaped = 0.5 + 0.5 * jnp.tanh(c/2)
    return 0.5 * sum(alpha * y**2) + sum(c_reshaped * jnp.log(jnp.cosh(W_diag * y + Bu + b)) / W_diag)


grad_state_kinetic_hamiltonian = jax.grad(kinetic_hamiltonian, argnums=(0))
grad_state_potential_hamiltonian = jax.grad(potential_hamiltonian, argnums=(0))
grad_parameters_potential_hamiltonian = jax.grad(potential_hamiltonian, argnums=(1, 2, 3, 4, 5))


def _apply_nonlinhru_leapfrog(nonlinhru_arg, carry_ini):
    """
    Apply the leapfrog integrator to evolve the NonlinHRU Hamiltonian system over time.
    
    This function implements a symplectic leapfrog integration scheme for Hamiltonian dynamics,
    which preserves the energy structure of the system. The integrator alternates between
    position and momentum updates using gradients of the kinetic and potential Hamiltonians.
    
    Args:
        nonlinhru_arg (tuple): Tuple containing system parameters:
            - Bu_elements (float): Input forcing sequence (B @ u), shape (L, P)
            - eps_nudging (float): Nudging perturbations for learning, shape (L, P)
            - W_diag (float): Diagonal state matrix, shape (P,)
            - b (float): Bias vector, shape (P,)
            - c (float): Scaling parameter vector, shape (P,)
            - alpha (float): Potential energy parameter, shape (1,) or scalar
            - step (float): Integration time-step size, shape (1,) or scalar
        carry_ini (tuple): Initial state (z_0, y_0):
            - z_0 (float): Initial momentum, shape (P,)
            - y_0 (float): Initial position, shape (P,)
    
    Returns:
        tuple: State sequences over time:
            - z (float): Momentum trajectory, shape (L, P)
            - y (float): Position trajectory, shape (L, P)
            - y_half (float): Intermediate position at half-steps, shape (L, P)
    """
    Bu_elements, eps_nudging, W_diag, b, c, alpha, step = nonlinhru_arg

    def state_update(carry, input_nudging):
        """Leapfrog integration step: position half-step, momentum full-step, position half-step."""
        z, y = carry
        Bu_element, eps_nudging = input_nudging

        # First half-step position update using kinetic Hamiltonian gradient
        y_half = y + 0.5 * step * grad_state_kinetic_hamiltonian(z)

        # Full-step momentum update using potential Hamiltonian gradient
        z -= step * grad_state_potential_hamiltonian(y, Bu_element, W_diag, b, c, alpha)

        # Second half-step position update and apply nudging to momentum
        y = y_half + 0.5 * step * grad_state_kinetic_hamiltonian(z)
        z += eps_nudging

        return (z, y), (z, y, y_half)

    # Scan over the input sequence to compute full trajectory
    _, (z, y, y_half) = jax.lax.scan(state_update, carry_ini, (Bu_elements, eps_nudging))

    return (z, y, y_half)


def apply_nonlinhru_leapfrog_bptt(vjp_arg, args=None):
    Bu_elements, W_diag, b, c, alpha, step, _ = vjp_arg

    carry_ini = (jnp.zeros_like(vjp_arg[2]), jnp.zeros_like(vjp_arg[2]))
    vjp_arg = (Bu_elements, jnp.zeros_like(Bu_elements), W_diag, b, c, alpha, step)

    z, y, _ = _apply_nonlinhru_leapfrog(vjp_arg, carry_ini)

    return y

# Function used when called outside of grad()
@eqx.filter_custom_vjp
def apply_nonlinhru_leapfrog(vjp_arg, args=None):
    Bu_elements, W_diag, b, c, alpha, step, _ = vjp_arg

    carry_ini = (jnp.zeros_like(vjp_arg[2]), jnp.zeros_like(vjp_arg[2]))
    vjp_arg = (Bu_elements, jnp.zeros_like(Bu_elements), W_diag, b, c, alpha, step)

    z, y, _ = _apply_nonlinhru_leapfrog(vjp_arg, carry_ini)

    return y

# Forward pass when called inside of grad()
@apply_nonlinhru_leapfrog.def_fwd
def fn_fwd(perturbed, vjp_arg, args):
    Bu_elements, W_diag, b, c, alpha, step, _ = vjp_arg

    carry_ini = (jnp.zeros_like(vjp_arg[2]), jnp.zeros_like(vjp_arg[2]))
    vjp_arg = (Bu_elements, jnp.zeros_like(Bu_elements), W_diag, b, c, alpha, step)

    z, y, _ = _apply_nonlinhru_leapfrog(vjp_arg, carry_ini)

    return y, (z[-1], y[-1])

# Backward pass when called inside of grad()
@apply_nonlinhru_leapfrog.def_bwd
def fn_bwd(residuals, grad_obj, perturbed, vjp_arg, args):
    Bu_elements, W_diag, b, c, alpha, step, epsilon = vjp_arg

    Bu_elements_reversed = Bu_elements[::-1, :]
    nudging_reversed = grad_obj[::-1, :]
    epsilons = jnp.array([+epsilon, -epsilon])

    def dynamics_and_grads(Bu_elements_reversed, nudging_reversed, W_diag, b, c, alpha, step, epsilon):
        eps_nudging = epsilon * nudging_reversed

        z_end, y_end = residuals
        carry_ini = (-z_end + eps_nudging[0], y_end)
        eps_nudging_reversed = jnp.concatenate([eps_nudging[1:], jnp.zeros_like(eps_nudging[:1])])

        # compute the dynamics
        nonlinhru_arg = (Bu_elements_reversed, eps_nudging_reversed, W_diag, b, c, alpha, step)
        z, y, y_half = _apply_nonlinhru_leapfrog(nonlinhru_arg, carry_ini)

        grads_t_hamiltonian_single_phase = jax.vmap(grad_parameters_potential_hamiltonian, in_axes=(0, 0, None, None, None, None))(
            y_half[:, :], Bu_elements_reversed, W_diag, b, c, alpha
        )

        # average gradient over time
        grads_hamiltonian_single_phase = [jnp.sum(grads, axis=0) if i != 0 else grads[::-1] for i, grads in enumerate(grads_t_hamiltonian_single_phase)]

        # add the null gradients
        full_grads_single_phase = (
            grads_hamiltonian_single_phase[0],
            grads_hamiltonian_single_phase[1],
            grads_hamiltonian_single_phase[2],
            grads_hamiltonian_single_phase[3],
            grads_hamiltonian_single_phase[4],
            jnp.zeros_like(step),
            jnp.zeros_like(epsilon),
        )
        return full_grads_single_phase

    # vmap the dynamics and the learning rule over the two echo passes
    full_grads = jax.vmap(dynamics_and_grads, in_axes=(None, None, None, None, None, None, None, 0))(
        Bu_elements_reversed, nudging_reversed, W_diag, b, c, alpha, step, epsilons
    )

    # do the contrastive gradient
    grads_eqprop = jax.tree.map(lambda x: - step * (x[0] - x[1]) / (2 * epsilon), full_grads)

    return grads_eqprop


class NonlinHRULayer(eqx.Module):
    W_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    b: jax.Array
    c: jax.Array
    alpha: jax.Array
    step: jax.Array
    learning_algorithm: str
    epsilon: float

    def __init__(self, ssm_size, H, learning_algorithm, epsilon, nbr_dt, *, key):

        B_key, C_key, D_key, W_key, step_key, key = jr.split(key, 6)
        self.W_diag = random.uniform(W_key, shape=(ssm_size,), minval=0.5, maxval=1.0)
        self.b = jnp.zeros((ssm_size,))
        self.c = random.uniform(key, shape=(ssm_size,), minval=-1.0, maxval=1.0)
        self.B = simple_uniform_init(B_key, shape=(ssm_size, H), std=1.0 / math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size), std=1.0 / math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.step = jnp.array([1e-4])
        self.alpha = random.uniform(step_key, shape=(1,), minval=0.1, maxval=1.0)
        self.learning_algorithm = learning_algorithm
        self.epsilon = epsilon

    def __call__(self, input_sequence):
        Bu_elements = jax.vmap(lambda u: self.B @ u)(input_sequence)
        
        c = jax.nn.sigmoid(self.c)
        vjp_arg = (Bu_elements, self.W_diag, self.b, c, self.alpha, self.step, self.epsilon)
        if self.learning_algorithm == "RHEL":
            y = apply_nonlinhru_leapfrog(vjp_arg, args=None)
        elif self.learning_algorithm == "BPTT":
            y = apply_nonlinhru_leapfrog_bptt(vjp_arg, args=None)
        ys_out = jax.vmap(lambda y: self.C @ y)(y)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys_out + Du


class NonlinHRUBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    ssm: NonlinHRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, ssm_size, H, learning_algorithm, epsilon, nbr_dt, drop_rate=0.05, *, key):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(input_size=H, axis_name="batch", channelwise_affine=False)
        self.ssm = NonlinHRULayer(
            ssm_size,
            H,
            learning_algorithm,
            epsilon,
            nbr_dt,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute NonlinearHRU block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class NonlinHRU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[NonlinHRUBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(self, num_blocks, N, ssm_size, H, output_dim, classification, output_step, learning_algorithm, epsilon, nbr_dt, *, key):

        linear_encoder_key, *block_keys, linear_layer_key = jr.split(key, num_blocks + 2)
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            NonlinHRUBlock(
                ssm_size,
                H,
                learning_algorithm,
                epsilon,
                nbr_dt,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute Nonlinear HSSM."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)  # Average over all timesteps
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
