from datetime import datetime
import numpy as np
import os
import warnings


class HybridDerivativeCalculator:
    """
           Initialize the Derivative Calculator with a function of a single parameter,
           a central value for evaluation, and parameters for derivative calculation
           and statistical analysis.

           Parameters:
           function (callable): The function whose derivative is to be calculated.
           x_center (float): The central value at which the derivative is evaluated.
           dx (float): The step size for derivative calculation.
           iterations (int): Number of iterations for derivative calculation.
           min_samples (int): Minimum number of samples to retain while adjusting the range.
           """
    def __init__(self, function, x_center, dx=0.01, min_samples=5, log_dir="logs", log_suffix=None):
        self.function = function
        self.x_center = x_center
        self.min_samples = min_samples
        self.dx = dx
        self.stem_deriv = None

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        # Generate timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d")
        if log_suffix:
            self.log_file = os.path.join(log_dir, f"stem_debug_{timestamp}_{log_suffix}.log")
        else:
            self.log_file = os.path.join(log_dir, f"stem_debug_{timestamp}.log")

    def _log_warning(self, message):
        warnings.warn(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def stem_method(self, include_zero=True, fallback=True, debug=True):
        """
        Calculate the stem derivative, with optional fallback to stencil if convergence fails.
        """
        percentages = [0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.1]
        if include_zero:
            try:
                eps = self.dx
                left = self.function(self.x_center - eps)
                right = self.function(self.x_center + eps)
                diff = abs(right - left)
                if diff > 2 * eps:
                    if debug:
                        self._log_warning("[stem_method] Function not smooth around x_center; excluding 0% offset.")
                    include_zero = False
            except (ValueError, TypeError, ArithmeticError):
                if debug:
                    self._log_warning("[stem_method] Error during smoothness check; excluding 0% offset.")
                include_zero = False

        if include_zero:
            percentages = [0.0] + percentages

        # Build x and y arrays
        if self.x_center == 0:
            x_values = np.array([p for p in percentages] + [-p for p in percentages])
        else:
            x_values = np.array([self.x_center * (1 + p) for p in percentages] +
                                [self.x_center * (1 - p) for p in percentages])
        y_values = np.array([self.function(x) for x in x_values])

        while len(x_values) >= self.min_samples:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fit = slope * x_values + intercept
            safe_y = np.where(y_values == 0, 1e-10, y_values)
            spread = np.abs((y_fit - y_values) / safe_y)
            max_spread = np.max(spread)

            if max_spread < 0.01:
                if debug:
                    self._log_warning(f"[stem_method] Converged with spread {max_spread:.4e}")
                return float(slope)
            else:
                x_values = x_values[1:-1]
                y_values = np.array([self.function(x) for x in x_values])

        if debug:
            self._log_warning("[stem_method] Did not converge; using fallback.")
        return self.five_point_stencil_method() if fallback else 0.0

    def _is_finite_and_differentiable(self, delta=1e-5, tol=1e-2):
        """Check if the function is finite and smoothly differentiable around x_center."""
        try:
            f0 = self.function(self.x_center)
            f_minus = self.function(self.x_center - delta)
            f_plus = self.function(self.x_center + delta)

            if not np.isfinite([f0, f_minus, f_plus]).all():
                return False

            left = (f0 - f_minus) / delta
            right = (f_plus - f0) / delta
            return abs(left - right) < tol

        except (ValueError, TypeError, ArithmeticError):
            return False

    def five_point_stencil_method(self):
        """
        Calculates the derivative of a function at a given central value using
        the five-point stencil method.
        The five-point stencil method is a finite difference method for numerical
        differentiation of a function. It uses five points around the central value
        to calculate the derivative.
        The equation for the five-point stencil method is given in
        https://en.wikipedia.org/wiki/Five-point_stencil.

        Returns:
            tuple: the five-point stencil derivative.
        """
        # Corrected five-point stencil coefficients for first derivative
        stencil_coeffs = np.array([1, -8, 0, 8, -1]) / (12.0 * self.dx)

        # Points around the central value for stencil calculation
        stencil_points = np.array([self.x_center + i * self.dx for i in range(-2, 3)])

        # Evaluate the function at these points and store in an array
        function_values = np.array([self.function(x) for x in stencil_points])

        # Calculate the derivative using the stencil coefficients
        derivative = np.dot(stencil_coeffs, function_values)

        return float(derivative)

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
            # Capture the current self.function in the closure
            def noisy_func(x, func=self.function):
                return func(x) + np.random.randn(1)[0] * noise_std

            # Choose the method for derivative calculation
            if method == 'stem':
                original_func = self.function
                self.function = noisy_func
                derivative = self.stem_method()
                self.function = original_func  # Restore original function
            elif method == 'five_point_stencil':
                original_func = self.function
                self.function = noisy_func
                derivative = self.five_point_stencil_method()
                self.function = original_func  # Restore original function
            else:
                raise ValueError("Invalid method. Choose 'stem' or 'five_point_stencil'.")

            derivatives.append(derivative)

        return derivatives
