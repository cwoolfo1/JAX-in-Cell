import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, config
from ._particles import rotation
from ._constants import epsilon_0, speed_of_light, elementary_charge, mass_electron, mass_proton
from ._particles import fields_to_particles_grid


def initialize_a(nx):
    """
    Initializes the scalar metric `a` for 2D General Relativity.

    This function creates a 1D array of shape (nx,) filled with ones, which represents
    the initial state of the scalar metric in the simulation.

    Args:
        nx (int): The number of grid points in the x-direction.

    Returns:
        jax.numpy.ndarray: A 1D array of shape (nx,) initialized to ones.
    """
    return jnp.ones(nx), jnp.ones(nx)

def energy_momentum_tensor(vs, ms):
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

    C = speed_of_light

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


def trace_energy_momentum_tensor(vs, ms):
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

    C = speed_of_light
    # Calculate the four-velocity for each particle
    # print(vs.shape)
    # print(ms.shape)
    four_velocity = jax.vmap(lambda v : jax.numpy.array([v[0], v[1], v[2], C / jax.numpy.sqrt(1 - (v[0]**2 + v[1]**2 + v[2]**2)/(C**2) )]), in_axes=(0))(vs)
    # v = (v0, v1, v2, gamma * C) defining four vector velocity
    # print(four_velocity.shape)
    trace = jax.vmap(lambda v, m: m*jnp.sum(jnp.square(v)), in_axes=(0,0))(four_velocity, ms)
    # compute the trace of outerproduct of the four velocity times the mass of the particle
    return jnp.sum(trace)


def solve_metric(a0, a1, lam, vs, ms, G, dx, dt):
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

    T = trace_energy_momentum_tensor(vs, ms)
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

def da_dx(a, dx):
    """
    Compute the spatial derivative of the scalar metric `a` fourier transform.
    This function calculates the derivative of the scalar metric `a` with respect to
    the spatial coordinate using the Fourier transform method.
    Args:
        a (ndarray): The scalar metric at the current time step.
        dx (float): The spatial resolution (grid spacing).
    Returns:
        ndarray: The spatial derivative of the scalar metric `a`.
    """

    a_k = jnp.fft.fft(a)
    # Fourier transform of the scalar metric a

    nx = a.shape[0]
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    # Create the wavevector in Fourier space

    a_k_dx = 1j * kx * a_k
    # Compute the Fourier transform of the derivative of a

    a_dx = jnp.fft.ifft(a_k_dx).real
    # Inverse Fourier transform to get the spatial derivative in real space

    return a_dx


def da_dt(a0, a1, dt):
    """
    Compute the time derivative of the scalar metric `a` using finite differences.
    
    This function calculates the time derivative of the scalar metric `a` at the current time step
    using the values from the previous two time steps.

    Args:
        a0 (ndarray): The scalar metric at the previous time step (t - dt).
        a1 (ndarray): The scalar metric at the current time step (t).
        dt (float): The time step for evolution.

    Returns:
        ndarray: The time derivative of the scalar metric `a`.
    """
    return (a1 - a0) / dt


@jit
def relativistic_boris_step(dt, xs_nplushalf, vs_n, q_ms, E_fields_at_x, B_fields_at_x, a1, a0, t, dx, grid, field_BC_left, field_BC_right):
    """
    This function performs one step of the Boris algorithm for particle motion. 
    The particle velocity is updated using the electric and magnetic fields at its position, 
    and the particle position is updated using the new velocity.

    Args:
        dt (float): Time step for the simulation.
        xs_nplushalf (array): The particle positions at the half-time step n+1/2, shape (N, 3).
        vs_n (array): The particle velocities at time step n, shape (N, 3).
        q_ms (array): The charge-to-mass ratio of each particle, shape (N, 1).
        E_fields_at_x (array): The interpolated electric field values at the particle positions, shape (N, 3).
        B_fields_at_x (array): The magnetic field values at the particle positions, shape (N, 3).

    Returns:
        tuple: A tuple containing:
            - xs_nplus3_2 (array): The updated particle positions at time step n+3/2, shape (N, 3).
            - vs_nplus1 (array): The updated particle velocities at time step n+1, shape (N, 3).
    """

    C = speed_of_light

    ################### Relativistic Corrections ###################
    a1_at_x = vmap(lambda x_n: fields_to_particles_grid(
    x_n, a1, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(xs_nplushalf)
    # interpolate the scalar metric `a1` at the particle positions

    da_dt_at_x = vmap(lambda x_n: fields_to_particles_grid(
        x_n, da_dt(a0, a1, dt), dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(xs_nplushalf)
    # Calculate the time derivative of the scalar metric `a` at the particle positions

    gamma = 1 / jnp.sqrt( -a1_at_x + jnp.sum(vs_n**2, axis=1) / C**2 / a1_at_x )
    # calculate the relativistic factor gamma at the particle positions

    relativistic_correction = - gamma * da_dt_at_x / (2 * a1_at_x) * (C**2 - jnp.sum(vs_n**2, axis=1) + C * jnp.sum(vs_n, axis=1) )
    # calculate the relativistic correction factor
    ################################################################

    # First half step update for velocity due to electric field
    vs_n_int = vs_n + (q_ms) * E_fields_at_x * dt / 2 * (-1 * a1_at_x) + relativistic_correction

    # Apply the Boris rotation step for the magnetic field
    vs_n_rot = vmap(lambda B_n, v_n, q_m: rotation(dt, B_n, v_n, q_m))(B_fields_at_x, vs_n_int, q_ms[:, 0])

    # Second half step update for velocity due to electric field
    vs_nplus1 = vs_n_rot + (q_ms) * E_fields_at_x * dt / 2 * (-1 * a1_at_x) + relativistic_correction
    # Update the particle velocities at time step n+1

    # Update the particle positions using the new velocities
    xs_nplus3_2 = xs_nplushalf + dt * vs_nplus1

    return xs_nplus3_2, vs_nplus1
    # vs_nplus1 = vs_n + (q_ms) * E_fields_at_x * dt
    # xs_nplus1 = xs_nplushalf + dt * vs_nplus1
    # return xs_nplus1, vs_nplus1