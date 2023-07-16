
import numpy as np

class AmericanOptionPricer:
    def __init__(self, N, m, T, basis_functions):
        self.N = N
        self.m = m
        self.T = T
        self.basis_functions = basis_functions

    def generate_stock_paths(self):
        # Generate N independent stock paths
        paths = np.zeros((self.N, self.m + 1))
        dt = self.T / self.m

        for i in range(self.N):
            paths[i, 0] = initial_stock_price  # Set initial stock price
            for j in range(self.m):
                # Simulate stock price evolution using your preferred method
                paths[i, j + 1] = simulate_next_stock_price(paths[i, j], dt)

        return paths

    def compute_option_price(self):
        paths = self.generate_stock_paths()

        # Step 2: Set terminal values
        terminal_values = g(paths[:, -1])

        # Step 3: Backward calculation
        for i in range(self.m - 1, 0, -1):
            current_paths = paths[:, i]
            continuation_values = np.zeros_like(current_paths)
            exercise_values = np.zeros_like(current_paths)

            for j in range(self.N):
                if g(current_paths[j]) > 0:  # Check if option is in the money
                    continuation_values[j] = self.compute_continuation_value(current_paths[j])
                    exercise_values[j] = g(current_paths[j])

            # Update option values
            terminal_values = np.where(exercise_values >= continuation_values, exercise_values, terminal_values)

        return np.mean(terminal_values)

    def compute_continuation_value(self, S):
        X = np.vstack([basis_func(S) for basis_func in self.basis_functions]).T
        return np.dot(X, self.optimal_coefficients)

    def solve_regression(self, paths, values):
        X = np.vstack([basis_func(paths) for basis_func in self.basis_functions]).T
        self.optimal_coefficients = np.linalg.inv(X.T @ X) @ X.T @ values

if __name__ == '__main__':

    # Usage example
    N = 10000  # Number of paths
    m = 100  # Number of exercise dates
    T = 1.0  # Time to maturity
    basis_functions = [lambda x: x, lambda x: x ** 2]  # Example of basis functions

    option_pricer = AmericanOptionPricer(N, m, T, basis_functions)
    option_price = option_pricer.compute_option_price()

    print("American Option Price:", option_price)
""" 

Note
that
this
code is a
simplified
implementation and assumes
that
you
have
defined
functions
`g`
for the exercise value and `simulate_next_stock_price` for 
simulating the next stock price.You would need to modify 
and customize the code to match your specific requirements and market model."""