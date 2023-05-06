import numpy as np
from numba import jit, vectorize, cuda
from syclbuffer import PyBuffer
from dpct import memory_scope
from dpct import kernel
import dpctl
from daal4py import daalinit, decomposition, linear_regression
import timeit


@vectorize(['float64(float64, float64, float64)'], target='cuda')
def gpu_black_scholes(s, k, t):
    d1 = (np.log(s / k) + (0.05 + 0.5 * 0.2 ** 2) * t) / (0.2 * np.sqrt(t))
    d2 = d1 - 0.2 * np.sqrt(t)
    return s * norm_cdf(d1) - k * np.exp(-0.05 * t) * norm_cdf(d2)

@jit(nopython=True)
def norm_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

@jit(nopython=True)
def cpu_black_scholes(num_simulations):
    # Define parameters
    s0 = 100.0
    k = 110.0
    t = 1.0
    r = 0.05
    sigma = 0.2
    dt = t / 252.0

    # Initialize arrays
    s = np.zeros(num_simulations)
    call_payoff = np.zeros(num_simulations)
    put_payoff = np.zeros(num_simulations)

    # Generate random numbers
    np.random.seed(42)
    z = np.random.normal(size=num_simulations)

    # Calculate stock prices
    for i in range(252):
        s *= np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate option payoffs
    call_payoff = np.maximum(s - k, 0)
    put_payoff = np.maximum(k - s, 0)

    # Calculate option prices
    call_price = np.exp(-r * t) * np.mean(call_payoff)
    put_price = np.exp(-r * t) * np.mean(put_payoff)

    return call_price, put_price


@kernel
def monte_carlo_simulations(num_simulations, num_steps, dt, s0, r, sigma, call_payoff, put_payoff):
    gid = kernel.get_global_id(0)
    lid = kernel.get_local_id(0)
    group_size = kernel.get_local_size(0)
    
    # Generate random numbers
    z = np.random.normal(size=(num_simulations, num_steps))

    # Calculate stock prices
    for i in range(num_steps):
        if lid == 0:
            s[gid, i + 1] = s[gid, i] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * z[gid, i])
        kernel.barrier()

    # Calculate option payoffs
    k = 110
    if lid == 0:
        call_payoff[gid] = np.maximum(s[gid, -1] - k, 0)
        put_payoff[gid] = np.maximum(k - s[gid, -1], 0)
    kernel.barrier()

    # Calculate option prices
    if lid == 0:
        call_price = np.exp(-r * dt * num_steps) * np.mean(call_payoff)
        put_price = np.exp(-r * dt * num_steps) * np.mean(put_payoff)
        return call_price, put_price
    kernel.barrier()
    option_price = exp(-r * T) * E[payoff]
call_price, put_price, lr_result = run_monte_carlo_simulations()
print("Call price:", call_price)
print("Put price:", put_price)
print("Linear regression result:", lr_result)