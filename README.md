# Cylindrical Storage Tank Optimization

Numerical optimization model designed to determine the minimum construction cost of a cylindrical storage tank under volume constraints. The implementation uses the Gradient Descent method coupled with a quadratic exterior penalty function and forward finite-difference numerical derivatives.

Developed for the Computational Modeling course (IPRJ / UERJ).

---

## Problem Formulation

The objective is to minimize total fabrication costs based on surface areas while satisfying a minimum required capacity.

### Cost Breakdown
* **Tank Base:** R$ 150.00 / m²
* **Lateral Wall:** R$ 100.00 / m²
* **Roof:** R$ 200.00 / m²

### Objective Function
The unconstrained cost function $f(r, h)$ sums the cost of base, wall, and roof areas:

$$f(r, h) = 150(\pi r^2) + 100(2\pi rh) + 200(\pi r^2) = 350\pi r^2 + 200\pi rh$$

### Constraints
The tank must satisfy a minimum volume capacity $V_{\min} = 500\text{ m}^3$. Expressed in standard form $g(r, h) \le 0$:

$$g(r, h) = 500 - \pi r^2 h \le 0$$

---

## Numerical Methods

### 1. Penalty Function Formulation
To solve the constrained formulation via unconstrained gradient methods, a quadratic exterior penalty is applied:

$$P(r, h) = f(r, h) + \mu \cdot \left[\max(0, g(r, h))\right]^2$$

* When $g(r, h) \le 0$ (feasible domain), the penalty term evaluates to zero.
* When $g(r, h) > 0$ (infeasible domain), a substantial numerical penalty controlled by parameter $\mu$ penalizes violations.

### 2. Forward Finite Differences
Gradients are approximated numerically rather than analytically, with step perturbation $\Delta = 10^{-5}$:

$$\frac{\partial P}{\partial r} \approx \frac{P(r + \Delta, h) - P(r, h)}{\Delta}$$

$$\frac{\partial P}{\partial h} \approx \frac{P(r, h + \Delta) - P(r, h)}{\Delta}$$

### 3. Gradient Descent Updates
Iterative parameters are updated along the negative gradient direction using step size $\alpha$:

$$r^{(k+1)} = r^{(k)} - \alpha \left.\frac{\partial P}{\partial r}\right\vert{}^{(k)}$$

$$h^{(k+1)} = h^{(k)} - \alpha \left.\frac{\partial P}{\partial h}\right\vert{}^{(k)}$$

---

## Getting Started

### Prerequisites
* Python 3.10+
* `numpy`, `matplotlib`

### Installation
```bash
git clone [https://github.com/](https://github.com/)<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
