import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Tank Base: R$ 150.00 / m²
# Lateral Wall: R$ 100.00 / m²
# Roof: R$ 200.00 / m²
# f ( r , h ) = 150 ( π r 2 ) + 100 ( 2 π r h ) + 200 ( π r 2 ) = 350 π r 2 + 200 π r h
# V min = 500  m³
# g ( r , h ) = 500 − π r 2 h ≤ 0
# P ( r , h ) = f ( r , h ) + μ ⋅ [ max ( 0 , g ( r , h ) ) ] 2
# (∂ P / ∂ r) ≈ (P ( r + Δ , h ) − P ( r , h )) /Δ
# (∂ P / ∂ h) ≈ (P ( r , h + Δ ) − P ( r , h )) /Δ
# r ( k + 1 ) = r ( k ) − α (∂ P / ∂ r) | ( k )
# h ( k + 1 ) = h ( k ) − α (∂ P / ∂ h) | ( k )

def cost_function(r, h):
    return 150 * (np.pi * r ** 2) + 100 * (2 * np.pi * r * h) + 200 * (np.pi * r ** 2)


def constraint(r, h):
    return 500 - np.pi * r ** 2 * h


def penalty_function(r, h, mu):
    g = constraint(r, h)
    return cost_function(r, h) + mu * (max(0, g)) ** 2


def partial_derivative_r(r, h, mu, delta):
    return ((penalty_function(r + delta, h, mu) - penalty_function(r, h, mu)) / delta)


def partial_derivative_h(r, h, mu, delta):
    return ((penalty_function(r, h + delta, mu) - penalty_function(r, h, mu)) / delta)


def gradient_r(r,h, mu, delta, alpha):
    r += - (alpha * partial_derivative_r(r,h, mu, delta))
    return r


def gradient_h(r,h, mu, delta, alpha):
    h += - (alpha * partial_derivative_h(r,h, mu, delta))
    return h


def random_r():
    r = rd.uniform(0.1, 10.0)  # Random radius between 1 and 10 meters
    return r


def random_h():
    h = rd.uniform(0.1, 20.0)  # Random height between 1 and 20 meters
    return h

def main():
    # Initial values
    r = random_r()  # Initial radius
    h = random_h()  # Initial height
    mu = 1e2 # Penalty parameter
    alpha = 1e-6  # Learning rate
    delta = 1e-5  # Small change for numerical derivative
    max_iterations = 1000000
    tolerance = 1e-7
    i = 0

    #i = iteration
    for i in range(max_iterations):

        # Update r and h using gradient
        r_new, h_new = gradient_r(r,h, mu, delta, alpha), gradient_h(r,h, mu, delta, alpha)

        # Check if r_new and h_new are valid
        r_new = max(0.1, r_new)
        h_new = max(0.1, h_new)

        # Check for convergence
        if abs(r_new - r) < tolerance and abs(h_new - h) < tolerance:
            break

        r, h = r_new, h_new

    #print(f"Number of iterations: {i}")
    #print(f"Optimized radius: {r:.8f} m")
    #print(f"Optimized height: {h:.8f} m")
    #print(f"Minimum cost: R$ {cost_function(r, h):.2f}")

    return cost_function(r, h)

def statistic_points_plot(cf):
    plt.plot(cf)
    plt.show()

# Initialize cf as a list to store the costs
cf = [0.0] * 30

for i in range(30):
    cf[i] = main()
    print(f"Iteration {i + 1}")
