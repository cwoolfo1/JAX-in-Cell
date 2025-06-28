import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, config
from ._particles import rotation
from ._constants import epsilon_0, speed_of_light, elementary_charge, mass_electron, mass_proton
from ._particles import fields_to_particles_grid
from ._boundary_conditions import field_2_ghost_cells


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
    return -1*jnp.ones(nx), -1*jnp.ones(nx), -1*jnp.ones(nx)

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

    four_velocity = jax.vmap(lambda v : jax.numpy.array([v[0], v[1], v[2], C * jax.numpy.sqrt(1 - (v[0]**2)/(C**2) )]))(vs, ms)
    # v = (v0, v1, v2, gamma * C) defining four vector velocity

    energy_momentum_tensor = jnp.asarray( shape = (4,4))
    # initialize the energy momentum tensor

    n_particles = vs.shape[0]

    for n in range(n_particles):
        for i in range(4):
            for j in range(4):
                energy_momentum_tensor[i,j] += ms[n] * four_velocity[n, i] * four_velocity[n, j]
    # populate the energy momentum tensor with particle 4 momentum


    return energy_momentum_tensor


def trace_energy_momentum_tensor(a1, vs, ms):
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
    # # Calculate the four-velocity for each particle
    four_velocity = jax.vmap(lambda v : jax.numpy.array([ C * jax.numpy.sqrt(1 - (v[0]**2 / C**2)),  v[0] ]), in_axes=(0))(vs)
    # v = (v0, gamma * C) defining four vector velocity
    trace = jax.vmap(lambda v, m: m*a1*v[0]**2 - m*v[1]**2 / a1, in_axes=(0,0))(four_velocity, ms)
    # compute the trace of outerproduct of the four velocity times the mass of the particle

    return jnp.sum(trace, axis=0)


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
    # d^2 a dx^2 + (1/a)'' = lam + 8piG/C^4*T
    # solve this diff eq.


    C = speed_of_light

    T = trace_energy_momentum_tensor(a1, vs, ms)
    # calculate the trace of the energy momentum tensor

    # nx = a1.shape[0]
    # # get the number of array points
    # kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    # # get the kx wavevector

    # a_k = jnp.fft.fft(a1)
    # # fourier transform alpha

    # a_k_xx = -kx**2 * a_k
    # # second derivative of a_k in fourier space
    # a_xx = jnp.fft.ifft(a_k_xx).real

    a_xx = ( jnp.roll(a1, shift=-1) + jnp.roll(a1, shift=1) - 2 * a1 ) / (dx**2)
    # compute the second derivative of a in real space using finite differences

    # d2_ainv_dt2 =  ( lam + 8*G*jnp.pi/C**4*T + a_xx )
    # # get the second derivative of a in time

    # a2_inv = 2*a1_inv - a0_inv + ( dt**2  * d2_ainv_dt2 )
    # # evolve the inverse of the metric using the differential equation

    # a2 = 1 / a2_inv
    # # compute the metric using its inverse
    # Ensure all calculations use 64-bit precision
    C = jnp.float64(C)
    G = jnp.float64(G)
    lam = jnp.float64(lam)
    dt = jnp.float64(dt)
    dx = jnp.float64(dx)
    a0 = a0.astype(jnp.float64)
    a1 = a1.astype(jnp.float64)
    a_xx = a_xx.astype(jnp.float64)
    T = jnp.float64(T)

    a2 = (2 * a1 - a0 - dt * a0 / a1 + dt**2 * a1**2 * (C**2 * lam + C**2*8*G*jnp.pi*T + C**2 * a_xx)) / (1 - dt / a1)




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

def da_dt(a0, a1, a2, dt):
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
    return (a2 - a0) / (2 * dt)

def wrap_around(i, n):
    """
    Wraps the index `i` around the size `n` to ensure it stays within bounds.

    This function handles negative indices and ensures that the index is always
    within the range [0, n-1].

    Args:
        i (int): The index to wrap.
        n (int): The size of the array.

    Returns:
        int: The wrapped index.
    """
    return lax.cond(i < 0, lambda _: i + n, lambda _: lax.cond(i >= n, lambda _: i - n, lambda _: i, None), None)

def interpolate_alpha(x_n, x_wind, field, dx):
    x = x_n[0]
    # Calculate the index of the field grid corresponding to the particle position
    n = field.shape[0]
    i = jnp.floor((x + x_wind/2) / dx).astype(int)

    # Linear interpolation
    i1 = i
    i2 = wrap_around(i + 1, n)

    delta_x = x -  (i * dx - x_wind/2)

    interp_value = field[i1] * (1 - delta_x / dx) + field[i2] * (delta_x / dx)
    return interp_value

def relativistic_electrostatic_step(xs_nplushalf, vs_n, q_ms, E_fields_at_x, a2, a1, a0, dx, dt, grid):
    """
    This function performs one step of the relativistic electrostatic particle motion update.
    It updates the particle positions and velocities based on the electric fields at their positions.

    Args:
        dt (float): Time step for the simulation.
        xs_nplushalf (array): The particle positions at the half-time step n+1/2, shape (N, 3).
        vs_n (array): The particle velocities at time step n, shape (N, 3).
        q_ms (array): The charge-to-mass ratio of each particle, shape (N, 1).
        E_fields_at_x (array): The interpolated electric field values at the particle positions, shape (N, 3).

    Returns:
        tuple: A tuple containing:
            - xs_nplus3_2 (array): The updated particle positions at time step n+3/2, shape (N, 3).
            - vs_nplus1 (array): The updated particle velocities at time step n+1, shape (N, 3).
    """

    C = speed_of_light


    x_wind = grid[-1] - grid[0]
    # Calculate the width of the grid in the x-direction

    # Interpolate the scalar metric `a1` at the particle positions
    a1_at_x = vmap(lambda x_n: interpolate_alpha(
        x_n, x_wind, a1, dx))(xs_nplushalf)

    da_dx_at_x = vmap(lambda x_n: interpolate_alpha(
        x_n, x_wind, da_dx(a1, dx), dx))(xs_nplushalf)
    # Calculate the spatial derivative of the scalar metric `a1` at the particle positions

    da_dt_at_x = vmap(lambda x_n: interpolate_alpha(
        x_n, x_wind, da_dt(a0, a1, a2, dt), dx))(xs_nplushalf)
    # Calculate the time derivative of the scalar metric `a1` at the particle positions

    vx_nplushalf = vs_n[:, 0]
    # Get the x-component of the particle velocities at time step n+1/2

    gamma = jnp.sqrt(1 - (vx_nplushalf/C)**2 )
    # Calculate the relativistic factor gamma at the particle positions

    vx_nplus1 = vx_nplushalf -1 * (q_ms[:,0]) * E_fields_at_x[:, 0] * a1_at_x * dt + \
        da_dx_at_x * dt * ( vx_nplushalf**2 / (2 * a1_at_x) / gamma   +   gamma * C**2 * a1_at_x / 2 ) - \
        da_dt_at_x * dt * ( gamma * vx_nplushalf / a1_at_x )
    # Update the x-component of the particle velocities at time step n+1
    
    # print(q_ms.shape)
    # print(E_fields_at_x.shape)
    # print(a1_at_x.shape)
    # print( ( -1 * q_ms[:,0] * E_fields_at_x[:, 0] * a1_at_x * dt ).shape)
    # print( ( da_dx_at_x * dt * ( vx_nplushalf**2 / (2 * a1_at_x) / gamma   +   gamma * C**2 * a1_at_x / 2 ) ).shape)
    # print( ( da_dt_at_x * dt * ( gamma * vx_nplushalf / a1_at_x ) ).shape)

    xs_nplus3_2 = xs_nplushalf[:, 0] + dt * vx_nplus1 / gamma
    # Update the particle positions using the new velocities

    # print( xs_nplus3_2.shape, vs_n.shape, vx_nplushalf.shape, vx_nplus1.shape, E_fields_at_x.shape, a1_at_x.shape, da_dx_at_x.shape, da_dt_at_x.shape)

    xs_nplus3_2 = jnp.stack((xs_nplus3_2, xs_nplushalf[:, 1], xs_nplushalf[:, 2]), axis=1)
    vs_nplus1 = jnp.stack((vx_nplus1, vs_n[:, 1], vs_n[:, 2]), axis=1)
    # Combine the updated x-component with the unchanged y and z components

    return xs_nplus3_2, vs_nplus1