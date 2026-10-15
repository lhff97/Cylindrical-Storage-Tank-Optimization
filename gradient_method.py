import numpy as np
import matplotlib.pyplot as plt
import random as rd

def cost_function(r, h):
    # Cost: 150(top) + 100(side) + 200(bottom)
    # The original equation had syntax errors. Corrected with proper '*' operators.
    return 150 * (np.pi * r ** 2) + 100 * (2 * np.pi * r * h) + 200 * (np.pi * r ** 2)

def constraint(r, h):
    # Volume constraint: V >= 500. Rewritten as 500 - V <= 0 for the penalty method.
    return 500 - (np.pi * (r ** 2) * h)

def penalty_function(r, h, mu):
    # Penalty function adding the constraint violation to the cost
    return cost_function(r, h) + mu * (max(0, constraint(r, h))) ** 2

def partial_derivative_r(r, h, mu, delta):
    # Numerical approximation of the partial derivative with respect to r
    return (penalty_function(r + delta, h, mu) - penalty_function(r, h, mu)) / delta

def partial_derivative_h(r, h, mu, delta):
    # Numerical approximation of the partial derivative with respect to h
    return (penalty_function(r, h + delta, mu) - penalty_function(r, h, mu)) / delta

def random_r():
    # Random initial guess for the radius
    return rd.uniform(0.1, 10.0)

def random_h():
    # Random initial guess for the height
    return rd.uniform(0.1, 20.0)

def gradient_method():
    r = random_r()
    h = random_h()

    # Hyperparameters for the gradient descent and penalty method
    mu = 10.0
    alpha = 0.01 
    delta = 1e-5
    max_iterations = 1000000
    tolerance = 1e-6

    for i in range(max_iterations):
        # Adjust learning rate (alpha) and penalty factor (mu) periodically
        if i % 500 == 0 and i > 0:
            alpha *= 0.98
            if constraint(r, h) > 0:
                mu *= 1.5

        # Calculate gradients
        dr = partial_derivative_r(r, h, mu, delta)
        dh = partial_derivative_h(r, h, mu, delta)

        # Gradient clipping to prevent exploding steps
        grad_norm = np.sqrt(dr**2 + dh**2)
        if grad_norm > 1.0:
            dr = dr / grad_norm
            dh = dh / grad_norm

        # Update rule
        r_new = r - (alpha * dr)
        h_new = h - (alpha * dh)

        # Prevent variables from becoming negative or zero
        r_new = max(0.1, r_new)
        h_new = max(0.1, h_new)

        # Check for convergence
        if abs(r_new - r) < tolerance and abs(h_new - h) < tolerance:
            break

        r, h = r_new, h_new
        
    return cost_function(r, h), r, h, i 

def plot_results(cf, r_vals, h_vals):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # --- TOP LEFT: Scatter of Costs ---
    runs = range(1, len(cf) + 1)
    ax1.plot(runs, cf, marker='o', linestyle='none', markersize=6, color='#1f77b4')
    
    min_val = np.min(cf)
    ax1.axhline(y=min_val, color='red', linestyle='--', linewidth=2, label=f"Mínimo Encontrado (R$ {min_val:,.2f})")
    
    ax1.set_title("Custo Final por Execução", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Número da Execução", fontsize=12)
    ax1.set_ylabel("Custo (R$)", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=11)

    # --- TOP RIGHT: Boxplot ---
    # showfliers=False disables automatic outliers so we can draw them manually
    box = ax2.boxplot(cf, patch_artist=True, widths=0.15, showmeans=True, showfliers=False,
                      meanprops={"marker":"D", "markerfacecolor":"#ff7f0e", "markeredgecolor":"black", "markersize":8})
    
    for patch in box['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.6)
    for median in box['medians']:
        median.set(color='red', linewidth=2)

    # Manually separate Outliers and Normal points using IQR limits
    q1 = np.percentile(cf, 25)
    q3 = np.percentile(cf, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cf_array = np.array(cf)
    is_outlier = (cf_array < lower_bound) | (cf_array > upper_bound)
    
    outliers = cf_array[is_outlier]
    normals = cf_array[~is_outlier]

    # Draw normal points with UNIFORM jitter to fill horizontal space
    if len(normals) > 0:
        x_jitter_normals = np.random.uniform(0.85, 1.15, size=len(normals))
        ax2.plot(x_jitter_normals, normals, 'k.', alpha=0.4, markersize=6, label="Pontos Individuais")

    # Draw outliers perfectly aligned to the center (x=1)
    if len(outliers) > 0:
        x_outliers_fixed = np.ones(len(outliers))
        ax2.plot(x_outliers_fixed, outliers, 'o', markerfacecolor='none', markeredgecolor='black', 
                 markersize=8, markeredgewidth=1.5, label="Outliers")

    median_val = np.median(cf)
    mean_val = np.mean(cf)
    max_val = np.max(cf)
    
    # Prevent visual break if the difference between max and min is close to zero
    if max_val - min_val < 1e-2:
        margin = max(1.0, min_val * 0.001)
        ax2.set_ylim(min_val - margin, max_val + margin)

    stats_text = (f"Média: R$ {mean_val:,.2f}\n"
                  f"Mediana: R$ {median_val:,.2f}\n"
                  f"Mínimo: R$ {min_val:,.2f}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.95, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='right', bbox=props)

    ax2.set_title("Distribuição dos Custos Finais", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Custo (R$)", fontsize=12)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Execuções do Algoritmo"], fontsize=12)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')

    # --- BOTTOM LEFT: Distribution of Radius (r) ---
    ax3.hist(r_vals, bins='auto', color='#2ca02c', edgecolor='black', alpha=0.7)
    mean_r = np.mean(r_vals)
    ax3.axvline(mean_r, color='red', linestyle='dashed', linewidth=2, label=f"Média: {mean_r:.2f}m")
    
    ax3.set_title("Distribuição do Raio (r)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Raio (Metros)", fontsize=12)
    ax3.set_ylabel("Frequência", fontsize=12)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(fontsize=11)

    # --- BOTTOM RIGHT: Distribution of Height (h) ---
    ax4.hist(h_vals, bins='auto', color='#9467bd', edgecolor='black', alpha=0.7)
    mean_h = np.mean(h_vals)
    ax4.axvline(mean_h, color='red', linestyle='dashed', linewidth=2, label=f"Média: {mean_h:.2f}m")
    
    ax4.set_title("Distribuição da Altura (h)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Altura (Metros)", fontsize=12)
    ax4.set_ylabel("Frequência", fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

def main():
    try:
        iterations = int(input("Digite o número de execuções: "))
    except ValueError:
        print("Entrada inválida. Usando 30 execuções por padrão.")
        iterations = 30

    # Initialize arrays
    cf = [0.0] * iterations
    r_vals = [0.0] * iterations
    h_vals = [0.0] * iterations
    
    print("\n" + "="*76)
    print(" INICIANDO OTIMIZAÇÃO ".center(76, "="))
    print("="*76)
    
    for i in range(iterations):
        cost, r, h, iters = gradient_method()
        
        cf[i] = cost
        r_vals[i] = r  
        h_vals[i] = h  
        
        print(f"Execução {i + 1:02d} | Iterações: {iters:<7d} | r: {r:5.2f} | h: {h:5.2f} | Custo: R$ {cost:,.2f}")

    print("\n" + "="*76)
    print(" ESTATÍSTICAS DE CONVERGÊNCIA ".center(76, "="))
    print("="*76)
    
    max_cf = np.max(cf)
    min_cf = np.min(cf)
    mean_cf = np.mean(cf)
    
    # Find the optimal r and h (those associated with the minimum cost)
    best_index = np.argmin(cf)
    best_r = r_vals[best_index]
    best_h = h_vals[best_index]
    
    print(f"Custo Máximo: R$ {max_cf:,.2f}")
    print(f"Custo Médio:  R$ {mean_cf:,.2f}")
    print(f"Custo Mínimo: R$ {min_cf:,.2f}  <-- MELHOR RESULTADO")
    print(f"Raio Ótimo (r): {best_r:.4f} m")
    print(f"Altura Ótima (h): {best_h:.4f} m")
    print("="*76 + "\n")

    plot_results(cf, r_vals, h_vals)

if __name__ == "__main__":
    main()
