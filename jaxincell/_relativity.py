import jax
import jax.numpy as jnp


def energy_momentum_tensor(vs, ms, C):
    """
    Computes the energy-momentum tensor for a system of particles.

    The energy-momentum tensor is calculated based on the four-velocity of 
    each particle and their respective masses. The four-velocity is derived 
    using the relativistic formula, incorporating the speed of light `C`.

    Args:
        vs (jax.numpy.ndarray): A 2D array of shape (n_particles, 3) representing 
            the 3-velocity components (v0, v1, v2) of each particle.
        ms (jax.numpy.ndarray): A 1D array of shape (n_particles,) representing 
            the masses of the particles.
        C (float): The speed of light constant.

    Returns:
        jax.numpy.ndarray: A 4x4 energy-momentum tensor representing the 
        contributions of all particles in the system.
    """

    four_velocity = jax.vmap(lambda v : jax.numpy.array([v[0], v[1], v[2], C / jax.numpy.sqrt(1 - (v[0]**2 + v[1]**2 + v[2]**2)/(C**2) )]))(vs, ms)
    # v = (v0, v1, v2, gamma * C) defining four vector velocity

    energy_momentum_tensor = jnp.asarray( shape = (4,4))
    # initialize the energy momentum tensor

    n_particles = vs.shape[0]

    for n in range(n_particles):
        for i in range(4):
            for j in range(4):
                energy_momentum_tensor[i,j] = ms[n] * four_velocity[n, i] * four_velocity[n, j]
    # populate the energy momentum tensor with particle 4 momentum


    return energy_momentum_tensor


def trace_energy_momentum_tensor(vs, ms, C):
    """
    Computes the trace of the energy-momentum tensor for a collection of particles.

    This function calculates the trace of the energy-momentum tensor by summing 
    the contributions from the outer product of the four-velocity vectors of 
    particles, scaled by their respective masses.

    Args:
        vs (jax.numpy.ndarray): A 2D array of shape (N, 3) representing the 
            3-velocity vectors of N particles. Each row corresponds to a 
            particle's velocity components (v0, v1, v2).
        ms (jax.numpy.ndarray): A 1D array of shape (N,) representing the masses 
            of the N particles.
        C (float): The speed of light constant.

    Returns:
        jax.numpy.ndarray: A scalar value representing the trace of the 
        energy-momentum tensor for the given particles.
    """

    four_velocity = jax.vmap(lambda v : jax.numpy.array([v[0], v[1], v[2], C / jax.numpy.sqrt(1 - (v[0]**2 + v[1]**2 + v[2]**2)/(C**2) )]))(vs, ms)
    # v = (v0, v1, v2, gamma * C) defining four vector velocity
    trace = jax.vmap(lambda v, m: m*jnp.sum(jnp.square(v)))(four_velocity, ms)
    # compute the trace of outerproduct of the four velocity times the mass of the particle
    return jnp.sum(trace)


def solve_metric(a0, a1, lam, vs, ms, C, G, dx, dt):
    """
    Evolve the scalar metric for 2D General Relativity using the energy-momentum tensor.

    This function solves a differential equation for the scalar metric `a` in 2D General Relativity.
    It uses the Fourier transform to compute spatial derivatives and evolves the metric in time
    based on the given parameters.

    Args:
        a0 (ndarray): The scalar metric at the previous time step (t - dt).
        a1 (ndarray): The scalar metric at the current time step (t).
        lam (float): A constant parameter in the differential equation.
        vs (ndarray): Array of velocities contributing to the energy-momentum tensor.
        ms (ndarray): Array of masses contributing to the energy-momentum tensor.
        C (float): A constant related to the energy-momentum tensor calculation.
        G (float): The gravitational constant.
        dx (float): The spatial resolution (grid spacing).
        dt (float): The time step for evolution.

    Returns:
        ndarray: The scalar metric at the next time step (t + dt).
    """
    # d^2 a dx^2 + (1/a)'' = lam + 8piG*T
    # solve this diff eq.

    a0_inv = 1/a0
    a1_inv = 1/a1
    # compute the inverses of the metric

    T = trace_energy_momentum_tensor(vs, ms, C)
    # calculate the trace of the energy momentum tensor

    nx = a1.shape[0]
    # get the number of array points
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    # get the kx wavevector

    a_k = jnp.fft.fft(a1)
    # fourier transform alpha

    a_k_xx = -kx**2 * a_k
    # second derivative of a_k in fourier space
    a_xx = jnp.fft.ifft(a_k_xx)

    d2_ainv_dt2 = lam + 8*jnp.pi*T - a_xx
    # get the second derivative of a in time

    a2_inv = 2*a1_inv - a0_inv + ( dt**2  * d2_ainv_dt2 )
    # evolve the inverse of the metric using the differential equation

    a2 = 1 / a2_inv
    # compute the metric using its inverse


    return a2