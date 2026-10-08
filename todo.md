# To-Do: Tank Project Optimization

## 1. Make it "Prettier" (Clean Code and Syntax Corrections)
- [ ] **Fix invalid line breaks:** Join the mathematical operations that are broken across multiple lines in the function returns (e.g., `cost_function` and `constraint`).
- [ ] **Improve variable names:** Replace short names with more descriptive ones. For example, change `cf` in `main` to `costs` or `final_costs`. Change `iterations` in `main` to `num_runs` or `num_simulations` (to avoid confusion with the gradient iterations).
- [ ] **Standardization:** Ensure spacing follows the Python PEP 8 standard (spaces after commas, two blank lines between functions).

## 2. Documentation and Comments
- [ ] **Docstrings:** Add a brief explanation (between triple quotes `"""`) right below each function definition, explaining what it takes and what it returns.
- [ ] **Block comments:** In the main loop (`for i in range(max_iterations)`), add comments explaining the step-by-step: 1. Gradient calculation, 2. Hyperparameter updating, 3. Limit application (`max(0.1)`), 4. Convergence check.

## 3. Resolve `mu` and `alpha` (Numerical Stability)
- [ ] **Adjust initial values:** Start with a slightly larger `alpha` (e.g., `1e-5`) and a smaller `mu` (e.g., `1.0` or `10.0`). A gigantic initial `mu` makes the derivative explode on the very first iteration.
- [ ] **Dynamic scaling:** 
  - Make `alpha` decay smoothly (e.g., multiply by `0.95` every 1000 iterations).
  - Make `mu` increase gradually at the same rate (e.g., multiply by `1.5` or `2.0`). This allows the gradient to find the sweet spot before the constraint wall becomes too "hard".
- [ ] **Implement Gradient Clipping (Optional, but recommended):** Create a small function or use native logic to prevent the values of the `partial_derivative_r` and `partial_derivative_h` variables from exceeding a limit (e.g., between `-10000` and `10000`). This prevents the radius and height from shooting to "infinity".

## 4. Charts (Matplotlib)
- [ ] **Cost Histogram:** Instead of just a linear `plt.plot()` that looks like an electrocardiogram, create a `plt.hist()` of the final costs array to see where the algorithm converges most often.
- [ ] **Scatter Plot:** Create a Radius vs. Height plot (`plt.scatter(radii_list, heights_list)`) to visualize if the solutions are converging to similar tank proportions.
- [ ] **Visual formatting:** Add titles (`plt.title`), axis labels (`plt.xlabel`, `plt.ylabel`), and grids (`plt.grid()`) to make the charts look like a professional report.

## 5. Better Statistical Analysis
- [ ] **Save dimensions history:** In `main()`, besides creating a list/array for the costs (`cf`), create arrays to save the final radius and final height for each simulation.
- [ ] **New metrics:** Import numpy to calculate and display in the final print:
  - Cost standard deviation (`np.std`).
  - Cost median (`np.median`).
- [ ] **Success rate:** Calculate the final volume of each simulation using the obtained radii and heights ($V = \pi r^2 h$) and count how many simulations actually hit the $V \ge 500$ target. Display this percentage at the end.
- [ ] **Highlight the best solution:** Identify the list index with the lowest cost and print a detailed message showing: "Best cost: $ X | Radius: Y m | Height: Z m".
