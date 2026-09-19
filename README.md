# Cylindrical Storage Tank Optimization

Numerical optimization model designed to determine the minimum construction cost of a cylindrical storage tank under volume constraints. The implementation uses the Gradient Descent method coupled with a quadratic exterior penalty function and forward finite-difference numerical derivatives.

Developed for the Computational Modeling course (IPRJ / UERJ)[cite: 1].

---

## Problem Formulation

The objective is to minimize total fabrication costs based on surface areas while satisfying a minimum required capacity[cite: 1].

### Cost Breakdown
* **Tank Base:** R$ 150.00 / m²[cite: 1]
* **Lateral Wall:** R$ 100.00 / m²[cite: 1]
* **Roof:** R$ 200.00 / m²[cite: 1]

### Objective Function
The unconstrained cost function $f(r, h)$ sums the cost of base, wall, and roof areas[cite: 1]:

$$f(r, h) = 150(\pi r^2) + 100(2\pi rh) + 200(\pi r^2) = 350\pi r^2 + 200\pi rh$$
[cite: 1]

### Constraints
The tank must satisfy a minimum volume capacity $V_{\min} = 500\text{ m}^3$[cite: 1]. Expressed in standard form $g(r, h) \le 0$[cite: 1]:

$$g(r, h) = 500 - \pi r^2 h \le 0$$
[cite: 1]

---

## Numerical Methods

### 1. Penalty Function Formulation
To solve the constrained formulation via unconstrained gradient methods, a quadratic exterior penalty is applied[cite: 1]:

$$P(r, h) = f(r, h) + \mu \cdot \left[\max(0, g(r, h))\right]^2$$
[cite: 1]

* When $g(r, h) \le 0$ (feasible domain), the penalty term evaluates to zero[cite: 1].
* When $g(r, h) > 0$ (infeasible domain), a substantial numerical penalty controlled by parameter $\mu$ penalizes violations[cite: 1].

### 2. Forward Finite Differences
Gradients are approximated numerically rather than analytically, with step perturbation $\Delta = 10^{-5}$[cite: 1]:

$$\frac{\partial P}{\partial r} \approx \frac{P(r + \Delta, h) - P(r, h)}{\Delta}$$
[cite: 1]

$$\frac{\partial P}{\partial h} \approx \frac{P(r, h + \Delta) - P(r, h)}{\Delta}$$
[cite: 1]

### 3. Gradient Descent Updates
Iterative parameters are updated along the negative gradient direction using step size $\alpha$[cite: 1]:

$$r^{(k+1)} = r^{(k)} - \alpha \left.\frac{\partial P}{\partial r}\right\vert{}^{(k)}$$
[cite: 1]

$$h^{(k+1)} = h^{(k)} - \alpha \left.\frac{\partial P}{\partial h}\right\vert{}^{(k)}$$
[cite: 1]

---

## Getting Started

### Prerequisites
* Python 3.10+ (or compatible runtime)
* `numpy`, `matplotlib`

### Installation
```bash
git clone [https://github.com/](https://github.com/)<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
