from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random
from jax.tree_util import Partial



def simple_uniform_init(rng, shape, std=1.0):
    weights = random.uniform(rng, shape) * 2.0 * std - std
    return weights


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def apply_lin_hru(A_diag, B, C, input_sequence, step):
    """Compute the linear HRU IMEX response for an input sequence.
    Args:
        A_diag (float): diagonal state matrix, shape (P,)
        B (complex): input matrix, shape (P, H)
        C (complex): output matrix, shape (H, P)
        input_sequence (float): input sequence of features, shape (L, H)
        step (float): discretization time-step $\Delta_t$, shape (P,)
    Returns:
        outputs (float): the SSM outputs (LinHRU_IMEX layer preactivations), shape (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag) - (step**2.0) * A_diag / 2.0
    B_ = -1.0 * step * A_diag
    C_ = step * (jnp.ones_like(A_diag) - (step**2.0) * A_diag / 4.0)
    D_ = jnp.ones_like(A_diag) - (step**2.0) * A_diag / 2.0

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0) / 2.0
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    ys = xs[:, A_diag.shape[0] :]


    return jax.vmap(lambda x: (C @ x).real)(ys)


def apply_lin_hru_internal_with_initial_state_and_nudging(lin_hru_arg, args):
    """Compute the full state sequence of the linear HRU.
    
    Args:
        lin_hru_arg (tuple): Tuple containing:
            - A_diag (float): diagonal state matrix, shape (P,)
            - F (float): preprocessed input, nudging and initial state, shape (L, 2*P)
            - step (float): discretization time-step $\Delta_t$, shape (P,)
        args: Additional arguments (currently unused)
    
    Returns:
        xs (float): Full state sequence containing momentum and position components, shape (L, 2*P)
                   First P elements are momentum (z), last P elements are position (y)
    """
    A_diag, F, step = lin_hru_arg

    A_ = jnp.ones_like(A_diag) - (step**2.0) * A_diag / 2.0
    B_ = -1.0 * step * A_diag
    C_ = step * (jnp.ones_like(A_diag) - (step**2.0) * A_diag / 4.0)
    D_ = jnp.ones_like(A_diag) - (step**2.0) * A_diag / 2.0

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((F.shape[0], 4 * A_diag.shape[0]))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    return xs


def Hamiltonian(z, y, u, A_diag, B, step):
    """
    Compute the Hamiltonian of the LinHRU system.

    Args:
        z (float): first component of the state of the system (momentum), shape (P,)
        y (float): second component of the state of the system (position), shape (P,)
        u (float): input of the system, shape (H,)
        A_diag (float): diagonal state matrix, shape (P,)
        B (float): input matrix, shape (P, H)
        step (float): discretization time-step, shape (P,)
    Returns:
        hamiltonian (float): scalar real value representing the Hamiltonian energy

    """
    hamiltonian = 0.5 * sum(A_diag * step * y**2) + 0.5 * sum(step * z**2) - sum((B @ u) * y * step)

    return hamiltonian


def Hamiltonian_complex(z, y, u, A_diag, B, step):
    """
    Compute the Hamiltonian of the LinHRU system with complex state variables.

    Args:
        z (complex): first component of the state of the system (momentum), shape (P,)
        y (complex): second component of the state of the system (position), shape (P,)
        u (float): input of the system, shape (H,)
        A_diag (float): diagonal state matrix, shape (P,)
        B (complex): input matrix, shape (P, H)
        step (float): discretization time-step, shape (P,)
    Returns:
        hamiltonian (float): scalar real value representing the Hamiltonian energy

    """
    hamiltonian = 0.5 * sum(A_diag * step * y**2) + 0.5 * sum(step * z**2) - sum((B @ u) * y * step).real
    return hamiltonian


def grad_Hamiltonian(complex_ssm, z, y, u, A_diag, B, step):
    """
    Compute gradients of the Hamiltonian with respect to system parameters.
    
    This function handles gradient computation for both real and complex state space models.
    For complex SSMs, it works around JAX's limitation that holomorphic gradients require
    complex inputs and outputs by converting real inputs to complex, computing holomorphic
    gradients, and extracting real components from the result.
    
    Args:
        complex_ssm (bool): Whether to use complex SSM formulation
        z (float or complex): First component of state (momentum), shape (P,)
        y (float or complex): Second component of state (position), shape (P,)
        u (float): Input to the system, shape (H,)
        A_diag (float): Diagonal state matrix, shape (P,)
        B (complex or float): Input matrix, shape (P, H)
        step (float): Discretization time-step, shape (P,)
    
    Returns:
        tuple: Gradients with respect to (u, A_diag, B, step):
            - grad_u (float): Gradient w.r.t. input, shape (H,)
            - grad_A_diag (float): Gradient w.r.t. A_diag, shape (P,)
            - grad_B (complex or float): Gradient w.r.t. B, shape (P, H)
            - grad_step (float): Gradient w.r.t. step, shape (P,)
    """
    if complex_ssm:
        compute_grad_complex = jax.grad(Hamiltonian_complex, argnums=(2, 3, 4, 5), holomorphic=True)
        A_diag_complex = A_diag + jnp.zeros_like(A_diag) * 1j
        step_complex = step + jnp.zeros_like(step) * 1j
        u_complex = u + jnp.zeros_like(u) * 1j

        grad_complex = compute_grad_complex(z, y, u_complex, A_diag_complex, B, step_complex)
        grad_decomplexified = (
            grad_complex[0].real,
            grad_complex[1].real,
            grad_complex[2],
            grad_complex[3].real,
        )
        return grad_decomplexified
    else:
        return jax.grad(Hamiltonian, argnums=(2, 3, 4, 5))(z, y, u, A_diag, B, step)


@eqx.filter_custom_vjp
def apply_linhru_internal_imex_with_initial_state(vjp_arg, args):
    A_diag, B, _, input_sequence, _, step, _ = vjp_arg
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0) / 2.0
    F = jnp.hstack((F1, F2))
    lin_hru_arg = (A_diag, F, step)
    xs = apply_lin_hru_internal_with_initial_state_and_nudging(lin_hru_arg, args)
    ys = xs[:, vjp_arg[0].shape[0] :]
    return ys


@apply_linhru_internal_imex_with_initial_state.def_fwd
def fn_fwd(perturbed, vjp_arg, args):
    A_diag, B, _, input_sequence, _, step, _ = vjp_arg
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0) / 2.0
    F = jnp.hstack((F1, F2))
    lin_hru_arg = (A_diag, F, step)
    xs = apply_lin_hru_internal_with_initial_state_and_nudging(lin_hru_arg, args)
    xs_end = xs[-1]
    ys = xs[:, A_diag.shape[0] :]
    return ys, xs_end


# TODO: clean the code below
@apply_linhru_internal_imex_with_initial_state.def_bwd
def fn_bwd(residuals, grad_obj, perturbed, vjp_arg, args):
    A_diag, B, C, input_sequence, x_ini, step, epsilon = vjp_arg
    complex_ssm = args

    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)[::-1, :]
    # + Sigma_x Nabla_x L when the cost is only on the position
    nudge = grad_obj[::-1, :]
    epsilons = jnp.array([+epsilon, -epsilon])
    # momentum reversal
    residuals_reversed = jnp.concatenate((-residuals[: A_diag.shape[0]], residuals[A_diag.shape[0] :]), axis=0)

    def wrapped_grad_hamiltonian(x, y, input_sequence, A_diag, B, step):
        return grad_Hamiltonian(complex_ssm, x, y, input_sequence, A_diag, B, step)


    # vmap the dynamics and the learning rule

    # dynamics: for computing the gradients
    def dynamics_and_grads(A_diag, B, C, input_sequence, Bu_elements, x_ini, step, args, nudge, residuals_reversed, epsilon):

        # perturbation already at the begining
        # shift the elements to make place for the initial state and add the nudge at the beginning
        Bu_elements = jnp.concatenate([jnp.zeros_like(Bu_elements[:1]), Bu_elements])  # + jnp.concatenate([epsilon * nudge, jnp.zeros_like(Bu_elements[:1])])=

        F1 = Bu_elements * step + jnp.concatenate([epsilon * nudge, jnp.zeros_like(Bu_elements[:1])])
        F2 = Bu_elements * (step**2.0) / 2.0
        F = jnp.hstack((F1, F2))
        F = F.at[0].add(residuals_reversed)

        # compute the dynamics
        lin_hru_arg = (A_diag, F, step)
        xs = apply_lin_hru_internal_with_initial_state_and_nudging(lin_hru_arg, args)

        # compute the intermediate Hamiltonian gradients
        def get_halfway(xs, step):
            y = xs[A_diag.shape[0] :]
            z = xs[: A_diag.shape[0]]
            y_half = y + step * z / 2.0
            return y_half

        y_intermediate = jax.vmap(get_halfway, in_axes=(0, None))(xs[:-1], step)

        # put to zero the last half of the first dim
        # y_intermediate = jnp.concatenate([y_intermediate[ : y_intermediate.shape[0] // 2], jnp.zeros_like(y_intermediate[y_intermediate.shape[0] // 2 :])])

        grads_t_halfway = jax.vmap(wrapped_grad_hamiltonian, in_axes=(0, 0, 0, None, None, None))(
            xs[:-1, : A_diag.shape[0]], y_intermediate, input_sequence[::-1], A_diag, B, step
        )

        grads_input = grads_t_halfway[0][::-1]
        # grads_input = grads_input.at[600:].set(jnp.zeros_like(grads_input[600:]))

        # learning rule: for updating the parameters
        # grads per time step (vmapped)
        # grads per parameter (summed over time steps)
        grads_t_hamiltonian_single_phase = jax.vmap(wrapped_grad_hamiltonian, in_axes=(0, 0, 0, None, None, None))(
            xs[1:, : A_diag.shape[0]], xs[1:, A_diag.shape[0] :], input_sequence[::-1], A_diag, B, step
        )
        # TODO: parallelize this
        grads_hamiltonian_single_phase = [jnp.sum(grads, axis=0) if i != 0 else grads[::-1] for i, grads in enumerate(grads_t_hamiltonian_single_phase)]

        # add the null gradients
        full_grads_single_phase = (
            grads_hamiltonian_single_phase[1],
            grads_hamiltonian_single_phase[2],
            jnp.zeros_like(C),
            grads_input,
            jnp.zeros_like(x_ini),
            grads_hamiltonian_single_phase[3],
            jnp.zeros_like(epsilon),
        )
        return full_grads_single_phase, xs, F

    full_grads, xs, F = jax.vmap(dynamics_and_grads, in_axes=(None, None, None, None, None, None, None, None, None, None, 0))(
        A_diag, B, C, input_sequence, Bu_elements, x_ini, step, args, nudge, residuals_reversed, epsilons
    )

    # do the contrastive gradient
    grads_eqprop = jax.tree_map(lambda x: - (x[0] - x[1]) / (2 * epsilon), full_grads)

    return grads_eqprop


class LinHRULayer(eqx.Module):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    learning_algorithm: str
    epsilon: float
    complex_ssm: bool
    train_steps: bool

    def __init__(self, ssm_size, H, learning_algorithm, epsilon, complex_ssm, train_steps, nbr_dt, *, key):

        B_key, C_key, D_key, A_key, step_key, key = jr.split(key, 6)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.B = simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1.0 / math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1.0 / math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key, shape=(ssm_size,))
        self.learning_algorithm = learning_algorithm
        self.epsilon = epsilon
        self.complex_ssm = complex_ssm
        self.train_steps = train_steps


    def __call__(self, input_sequence):
        A_diag = nn.relu(self.A_diag)

        if self.complex_ssm:
            B_complex = self.B[..., 0] + 1j * self.B[..., 1]
            C_complex = self.C[..., 0] + 1j * self.C[..., 1]
        else:
            B_complex = self.B[..., 0]
            C_complex = self.C[..., 0]

        steps = nn.sigmoid(self.steps)
        if self.train_steps is False:
            steps = jax.lax.stop_gradient(steps)
        input_sequence_perturb = input_sequence

        if self.learning_algorithm == "BPTT":
            ys_out = apply_lin_hru(A_diag, B_complex, C_complex, input_sequence_perturb, steps)
        elif self.learning_algorithm == "RHEL":
            vjp_args = (A_diag, B_complex, C_complex, input_sequence_perturb, jnp.zeros_like(A_diag), steps, self.epsilon)
            args = self.complex_ssm
            # only the internal computation
            ys = apply_linhru_internal_imex_with_initial_state(vjp_args, args)
            # output of the LinHRU layer
            ys_out = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        else:
            print("Learning algorithm type not implemented")

        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys_out + Du


class LinHRUBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    ssm: LinHRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, ssm_size, H, learning_algorithm, epsilon, complex_ssm, train_steps, nbr_dt, drop_rate=0.05, *, key):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(input_size=H, axis_name="batch", channelwise_affine=False)
        self.ssm = LinHRULayer(
            ssm_size,
            H,
            learning_algorithm,
            epsilon,
            complex_ssm,
            train_steps,
            nbr_dt,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute LinHRU block."""
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


class LinHRU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinHRUBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(self, num_blocks, N, ssm_size, H, output_dim, classification, output_step, learning_algorithm, epsilon, complex_ssm, train_steps, nbr_dt, *, key):

        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(key, num_blocks + 3)
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            LinHRUBlock(
                ssm_size,
                H,
                learning_algorithm,
                epsilon,
                complex_ssm,
                train_steps,
                nbr_dt,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute Linear HSSM."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
