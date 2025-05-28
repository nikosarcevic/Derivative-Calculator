from datetime import datetime
import os
import warnings
import numba as nb
import numpy as np


class HybridDerivativeCalculator:
    """
           Initialize the Derivative Calculator with a function of a single parameter,
           a central value for evaluation, and parameters for derivative calculation
           and statistical analysis.

           Parameters:
           function (callable): The function whose derivative is to be calculated.
           central_value (float): The central value at which the derivative is evaluated.
           stem_precision (float): The precision for convergence in the stem method.
           stencil_stepsize (float): The step size for derivative calculation.
           min_samples (int): Minimum number of samples to retain while adjusting the range.
           """
    def __init__(self,
                 function,
                 central_value,
                 stem_precision=0.05,
                 stencil_stepsize=0.01,
                 derivative_order=1,
                 min_samples=5,
                 log_dir="logs",
                 log_suffix=None):
        self.function = function
        self.central_value = central_value
        self.min_samples = min_samples
        self.stem_precision = stem_precision
        self.stencil_stepsize = stencil_stepsize
        self.derivative_order = derivative_order
        self.stem_deriv = None
        self.percentages = np.array([0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.1])

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        # Generate timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d")
        if log_suffix:
            self.log_file = os.path.join(log_dir, f"stem_debug_{timestamp}_{log_suffix}.log")
        else:
            self.log_file = os.path.join(log_dir, f"stem_debug_{timestamp}.log")

    def stem_method(self, include_zero=True, fallback=True, debug=True):
        """
        Estimate the 1st or 2nd order derivative using iterative polynomial fitting.

        Returns:
            float: Estimated derivative at central_value.
        """
        if self.derivative_order not in [1, 2]:
            raise ValueError("Only derivative_order=1 or 2 is supported.")

        percentages = np.insert(self.percentages, 0, 0.0) if include_zero else self.percentages.copy()
        if include_zero and not self._is_finite_and_differentiable():
            if debug:
                msg = "[stem_method] Function not smooth or differentiable at central_value; excluding 0% offset."
                warnings.warn(msg)
                self._log_debug_message(msg)

        # Build x and y arrays
        if self.central_value == 0:
            x_values = np.array([p for p in percentages] + [-p for p in percentages])
        else:
            x_values = np.array([self.central_value * (1 + p) for p in percentages] +
                                [self.central_value * (1 - p) for p in percentages])
        y_values = np.array([self.function(x) for x in x_values])

        while len(x_values) >= max(self.min_samples, self.derivative_order + 2):
            coeffs = polyfit(x_values, y_values, self.derivative_order)
            max_spread = compute_spread(x_values, y_values, coeffs)
            if max_spread < self.stem_precision:
                if debug:
                    msg = f"[stem_method] Converged with spread {max_spread:.4e}"
                    self._log_debug_message(msg)
                # Return appropriate derivative at central_value
                return poly_derivative(np.atleast_1d(self.central_value), coeffs, self.derivative_order)[0]

            x_values = x_values[1:-1]
            y_values = y_values[1:-1]

        if debug:
            self._log_debug_message("[stem_method] Did not converge; using fallback.")

        return self.five_point_stencil_method() if fallback else 0.0

    def five_point_stencil_method(self):
        """
        Calculates the derivative of a function at a given central value using
        the five-point stencil method.

        Supports first and second derivatives.

        Returns:
            float: the five-point stencil derivative.
        """
        h = self.stencil_stepsize
        x0 = self.central_value

        # Generate stencil points
        stencil_points = np.array([x0 + i * h for i in range(-2, 3)])
        function_values = np.vectorize(self.function)(stencil_points)

        # Coefficients for derivatives
        coeffs_dict = {
            1: np.array([1, -8, 0, 8, -1]) / (12.0 * h),
            2: np.array([-1, 16, -30, 16, -1]) / (12.0 * h ** 2),
        }

        coeffs = coeffs_dict.get(self.derivative_order)
        if coeffs is None:
            raise ValueError("Only derivative_order=1 or 2 is supported.")

        derivative = np.dot(coeffs, function_values)
        return float(derivative)

    def _log_debug_message(self, message, debug=True):
        if debug:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def _is_finite_and_differentiable(self, delta=1e-5, tol=1e-2):
        """Check if the function is finite and smoothly differentiable around central_value."""
        try:
            f0 = self.function(self.central_value)
            f_minus = self.function(self.central_value - delta)
            f_plus = self.function(self.central_value + delta)

            if not np.isfinite([f0, f_minus, f_plus]).all():
                return False

            left = (f0 - f_minus) / delta
            right = (f_plus - f0) / delta
            return np.all(np.abs(left - right) < tol)

        except (ValueError, TypeError, ArithmeticError):
            return False

    def calculate_derivatives_with_output_noise(self, method='stem', iterations=100, noise_std=0.01):
        """
        Calculates the derivatives of a function at a given central value with output noise.
        The "output noise" is added to the function values before derivative calculation.
        The noise is assumed to be Gaussian with zero mean and a given standard deviation.

        Parameters:
            method (str): The method to use for derivative calculation.
            iterations (int): Number of iterations for derivative calculation.
            noise_std (float): Standard deviation of the output noise.

        Returns:
            list: The list of derivatives calculated.
        """
        derivatives = []

        for _ in range(iterations):
            def noisy_func(x, func=self.function):
                return func(x) + np.random.randn() * noise_std

            original_func = self.function
            self.function = noisy_func
            derivative = np.nan  # Declare here for linter happiness

            try:
                try:
                    if method == 'stem':
                        derivative = self.stem_method()
                    elif method == 'five_point_stencil':
                        derivative = self.five_point_stencil_method()
                    else:
                        raise ValueError("Invalid method. Choose 'stem' or 'five_point_stencil'.")
                except Exception as e:
                    self._log_debug_message(f"[calculate_derivatives_with_output_noise] Derivative failed: {e}")
                    # derivative already set to nan
            finally:
                self.function = original_func

            derivatives.append(derivative)

@nb.njit
def poly_derivative(x_values, coeffs, deriv_order):
    mat = np.zeros((x_values.shape[0], len(coeffs)))
    for i in range(mat.shape[1]):
        if i < deriv_order:
            mat[:, i] = 0
        else:
            mat[:, i] = x_values**(i - deriv_order) * np.prod(i - np.arange(0, deriv_order))
    return mat @ coeffs
    

@nb.njit
def poly_mat(x_values, deg):
    mat = np.zeros((x_values.shape[0], deg + 1))
    for i in range(mat.shape[1]):
        mat[:,i] = x_values**i
    return mat


@nb.njit
def polyfit(x_values, y_values, deg):
    mat = poly_mat(x_values, deg)
    coeffs, _, _, _ = np.linalg.lstsq(mat, y_values)
    return coeffs


@nb.njit
def compute_spread(x_values, y_values, fit_coeff):
    y_fit = poly_mat(x_values, len(fit_coeff) - 1) @ fit_coeff
    safe_y = np.maximum(1e-10, y_values)
    max_spread = np.max(np.abs((y_fit - y_values) / safe_y))
    return max_spread
