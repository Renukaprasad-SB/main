import numpy as np
import scipy.stats as si
import math 

# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type! Use 'call' or 'put'.")

# Merton Jump-Diffusion Model
def merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, N=50, option_type="call"):
    price = 0
    for k in range(N):
        r_adj = r - lam * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1)
        sigma_adj = np.sqrt(sigma ** 2 + k * sigma_j ** 2 / T)
        poisson_prob = (np.exp(-lam * T) * (lam * T) ** k) / math.factorial(k)
        price += poisson_prob * black_scholes(S * np.exp(k * mu_j), K, T, r_adj, sigma_adj, option_type)
    return price

# Example Usage:
S = 100    
K = 100   
T = 1     
r = 0.05   
sigma = 0.2  
lam = 0.1   
mu_j = -0.02 
sigma_j = 0.1 

print("Black-Scholes Call Price:", black_scholes(S, K, T, r, sigma, "call"))
print("Merton Jump-Diffusion Call Price:", merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="call"))
print("Black-Scholes Put Price:", black_scholes(S, K, T, r, sigma, "put"))
print("Merton Jump-Diffusion Put Price:", merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="put"))
