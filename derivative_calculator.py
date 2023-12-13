import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy as sp
import seaborn as sns


class DerivativeCalculator:
    def __init__(self, myfunc, x_center, dx=0.01, min_samples=5):
        """
        Initialize the Derivative Calculator with a function of a single parameter,
        a central value for evaluation, and parameters for derivative calculation
        and statistical analysis.

        Parameters:
        myfunc (callable): The function whose derivative is to be calculated.
        x_center (float): The central value at which the derivative is evaluated.
        dx (float): The step size for derivative calculation.
        iterations (int): Number of iterations for derivative calculation.
        min_samples (int): Minimum number of samples to retain while adjusting the range.
        """
        self.myfunc = myfunc
        self.x_center = x_center
        self.min_samples = min_samples
        self.dx = dx
        self.stem_deriv = None


    def stem_method(self):
        """
        Calculates the stem derivative of a function at a given central value.

        The "stem" derivative is based on a method developed by Camera et al. as described
        in the paper "SKA Weak Lensing III: Added Value of Multi-Wavelength Synergies for
        the Mitigation of Systematics" (https://arxiv.org/abs/1606.03451).
        A detailed description of the method can be found in Appendix B of the paper.

        Note that this method is not applicable for functions with a zero derivative at
        the central value. It also may require a relatively long time to converge for
        functions with a very small derivative at the central value.

        Additionally, this implementation of the stem method uses lambda functions, which
        may not be the most efficient way to implement it. If you intend to use this method
        within a larger framework (e.g., in a Fisher forecasting), you may consider
        modifying the implementation to avoid the use of lambda functions or to better
        integrate it into your specific framework.

        Returns:
        tuple: the stem derivative.
        """

        # The percentage values to use for the stem method
        # Note that this is an arbitrary choice and the values
        # can be changed as needed
        percentages = [0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05,
                       0.1]  # 0.625%, 1.25%, 1.875%, 2.5%, 3.75%, 5%, 10%
        stem_deriv = []  # List to store the stem derivative

        # Use a fixed range around zero for x values if the central value is zero
        if self.x_center == 0:
            x_values = np.array([p for p in percentages]
                                + [-p for p in percentages])
        # Use a fixed range around the central value for x values otherwise
        else:
            x_values = np.array([self.x_center * (1 + p) for p in percentages]
                                + [self.x_center * (1 - p) for p in percentages])
        # Evaluate the function at these x values
        y_values = np.stack([self.myfunc(x) for x in x_values], axis=0)

        # Fit a line to the data points and calculate the spread
        while len(x_values) >= self.min_samples:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fitted = slope * x_values + intercept
            spread = np.abs((y_fitted - y_values) / y_values)
            max_spread = np.max(spread)

            # If the spread is small enough, return the slope as the derivative
            # Also note that this criterium is an arbitrary choice
            # and the value can be changed as needed
            if max_spread < 0.01:
                stem_deriv.append(slope)
                break
            # Otherwise, remove the outlier point with the maximum spread
            else:
                x_values = x_values[1:-1]
                y_values = y_values[1:-1]

        return stem_deriv[0] if stem_deriv else 0  # Or return None

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
        function_values = np.array([self.myfunc(x) for x in stencil_points])

        # Calculate the derivative using the stencil coefficients
        derivative = np.dot(stencil_coeffs, function_values)

        return derivative

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
            # Capture the current self.myfunc in the closure
            def noisy_func(x, func=self.myfunc):
                return func(x) + np.random.randn(1)[0] * noise_std

            # Choose the method for derivative calculation
            if method == 'stem':
                original_func = self.myfunc
                self.myfunc = noisy_func
                derivative = self.stem_method()
                self.myfunc = original_func  # Restore original function
            elif method == 'five_point_stencil':
                original_func = self.myfunc
                self.myfunc = noisy_func
                derivative = self.five_point_stencil_method()
                self.myfunc = original_func  # Restore original function
            else:
                raise ValueError("Invalid method. Choose 'stem' or 'five_point_stencil'.")

            derivatives.append(derivative)

        return derivatives

    def plot_derivative_distributions(self,
                                      method1_data,
                                      method2_data,
                                      method1_name='Five-Point Stencil',
                                      method2_name='Stem Method',
                                      bins=20):
        """
        Plots histograms of derivative calculations for two methods side by side,
        and compares with numdifftools derivative calculation.

        Parameters:
        method1_data (list): The derivative data from the first method.
        method2_data (list): The derivative data from the second method.
        method1_name (str): The name of the first method.
        method2_name (str): The name of the second method.
        bins (int): Number of bins for the histogram.
        """
        plt.figure(figsize=(20, 10))  # Set the figure size
        lw = 2.5  # Line width
        alpha = 1  # Transparency

        # Set the global parameters for all plots
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15  # Set x-tick label font size
        plt.rcParams['ytick.labelsize'] = 15

        # Calculate derivative using numdifftools
        nd_derivative = nd.Derivative(self.myfunc)(self.x_center)

        # Plot for method 1
        plt.subplot(1, 2, 1)
        plt.hist(method1_data, bins=bins, color='hotpink', alpha=alpha)
        plt.axvline(np.mean(method1_data), color='k', linestyle=':', lw=lw, label=f'Mean: {np.mean(method1_data):.2f}')
        plt.axvline(np.median(method1_data), color='gray', linestyle='--', lw=lw,
                    label=f'Median: {np.median(method1_data):.2f}')
        plt.plot([], [], ' ', label=f"ND Derivative: {nd_derivative:.2f}")  # Empty plot for legend numdifftools
        plt.plot([], [], ' ', label=f"Variance: {np.var(method1_data):.2f}")  # Empty plot for legend variance
        plt.title(f"{method1_name} Derivative Distribution", fontsize=25)
        plt.xlabel("Derivative")
        plt.ylabel("Frequency")
        plt.legend(frameon=False, loc='upper left')

        # Plot for method 2
        plt.subplot(1, 2, 2)
        plt.hist(method2_data, bins=bins, color='yellowgreen', alpha=alpha)
        plt.axvline(np.mean(method2_data), color='k', linestyle=':', lw=lw, label=f'Mean: {np.mean(method2_data):.2f}')
        plt.axvline(np.median(method2_data), color='gray', linestyle='--', lw=lw,
                    label=f'Median: {np.median(method2_data):.2f}')
        plt.plot([], [], ' ', label=f"ND Derivative: {nd_derivative:.2f}")  # Empty plot for legend numdifftools
        plt.plot([], [], ' ', label=f"Variance: {np.var(method2_data):.2f}")  # Empty plot for legend variance
        plt.title(f"{method2_name} Derivative Distribution", fontsize=25)
        plt.xlabel("Derivative")
        plt.ylabel("Frequency")
        plt.legend(frameon=False, loc='upper left')
        # Adjust layout to ensure everything fits without clipping
        plt.tight_layout()
        # Save the figure
        plt.savefig("derivation_comparison_hist.png", dpi=300)

        plt.show()


    def demonstrate_stem_method(self, num_points=20, noise_std=0.01):
        """
        Demonstrates the stem method with fitting using noisy data points.
        It generates noisy data points, fits a line using the stem method,
        and plots the noisy data points and the fitted line.
        Please note that this is a simplified version  of the stem_method() function.

        Parameters:
        num_points (int): Number of data points to generate and fit.
        noise_std (float): Standard deviation of the noise to add to data points.

        Returns:
            plot: The plot of noisy data points and fitted line.
        """
        # Generate noisy data points
        x_values = np.linspace(self.x_center - 0.1, self.x_center + 0.1, num_points)
        y_values = [self.myfunc(x) + np.random.randn(1)[0] * noise_std for x in x_values]

        # Fit a line using the stem method
        slope, intercept, max_spread = self.stem_method_with_fitting(x_values, y_values)

        # Set the global parameters for all plots
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # Plot the noisy data points and fitted line
        plt.figure(figsize=(7, 5))
        plt.scatter(x_values, y_values, label='Noisy Data', color='purple', s=50)
        fit = slope * x_values + intercept
        plt.plot(x_values, fit, label='Fitted Line', color='yellowgreen', lw=3)
        plt.axvline(self.x_center, color='darkgray', linestyle='--', lw=2, label='Central value + noise')

        plt.title('Demonstration of Stem Method with Noisy Data', fontsize=15)
        plt.xlabel('$x$ (central value and stem values)', fontsize=12)
        plt.ylabel('$y$ (evaluated function values)', fontsize=12)
        plt.legend(fontsize=12, frameon=False)
        plt.xticks(fontsize=10)  # Adjust the font size for x-axis tick labels as needed
        plt.yticks(fontsize=10)
        # Adjust layout to ensure everything fits without clipping
        plt.tight_layout()
        # Save the figure
        plt.savefig("stem_demonstration.png", dpi=300)

        plt.show()

    def stem_method_with_fitting(self, x_values, y_values):
        """
        Calculates the stem method with fitting using provided data points.
        It fits a line to the data points and removes the outlier points iteratively.
        The fitting is done using numpy.polyfit.
        Other fitting methods can be used instead.

        Parameters:
        x_values (array-like): x values of data points.
        y_values (array-like): y values of data points.

        Returns:
        tuple: Slope and intercept of the fitted line.
        """
        # Initial fitting
        slope, intercept = np.polyfit(x_values, y_values, 1)
        y_fitted = slope * x_values + intercept
        spread = np.abs((y_fitted - y_values) / y_values)  # Spread of the fitted line
        max_spread = np.max(spread)  # Maximum spread

        # Refinement with removal of outliers and re-fitting
        while max_spread >= 0.01 and len(x_values) >= self.min_samples:
            # Find the index of the point with maximum spread
            max_spread_index = np.argmax(spread)

            # Remove the outlier point
            x_values = np.delete(x_values, max_spread_index)
            y_values = np.delete(y_values, max_spread_index)

            # Fit the line again
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fitted = slope * x_values + intercept
            spread = np.abs((y_fitted - y_values) / y_values)
            max_spread = np.max(spread)

        return slope, intercept, max_spread

    def plot_derivatives_boxplot(self, iterations=100, noise_std=0.01):
        """
        Calculates the derivatives of a function at a given central value with output noise.
        The "output noise" is added to the function values before derivative calculation.
        The noise is assumed to be Gaussian with zero mean and a given standard deviation.

        Parameters:
        iterations (int): Number of iterations for derivative calculation.
        noise_std (float): Standard deviation of the output noise.
        """
        stem_derivatives = []
        stencil_derivatives = []

        for _ in range(iterations):
            # Capture the current self.myfunc in the closure
            def noisy_func(x, func=self.myfunc):
                return func(x) + np.random.randn(1)[0] * noise_std

            # Calculate derivatives using the stem method
            original_func = self.myfunc
            self.myfunc = noisy_func
            stem_derivative = self.stem_method()
            self.myfunc = original_func  # Restore original function
            stem_derivatives.append(stem_derivative)

            # Calculate derivatives using the five-point stencil method
            original_func = self.myfunc
            self.myfunc = noisy_func
            stencil_derivative = self.five_point_stencil_method()
            self.myfunc = original_func  # Restore original function
            stencil_derivatives.append(stencil_derivative)

        # Create two subplots for each method
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # Set box colors
        stem_color = 'yellowgreen'
        stencil_color = 'hotpink'

        # Plot the box plots for Five-Point Stencil on top
        bp1 = axes[0].boxplot(stencil_derivatives, labels=['Five-Point Stencil'], vert=False,
                              patch_artist=True, boxprops=dict(facecolor=stencil_color),
                              medianprops=dict(color='black'), meanline=True)

        # Plot the box plots for Stem Method on the bottom
        bp2 = axes[1].boxplot(stem_derivatives, labels=['Stem Method'], vert=False,
                              patch_artist=True, boxprops=dict(facecolor=stem_color),
                              medianprops=dict(color='black'), meanline=False)

        # Add medians and variances to the legend
        stem_median = np.median(stem_derivatives)
        stencil_median = np.median(stencil_derivatives)
        stem_variance = np.var(stem_derivatives)
        stencil_variance = np.var(stencil_derivatives)

        axes[1].set_xlabel('Derivative', fontsize=12)
        # Adjust layout to ensure everything fits without clipping
        plt.tight_layout()
        # Save the figure
        plt.savefig("derivation_comparison_boxplot.png", dpi=300)

        plt.show()
