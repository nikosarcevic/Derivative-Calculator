import os

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np

from hybrid_derivative_calculator import HybridDerivativeCalculator


class DerivativePlotter:
    def __init__(self, function, x_center, plot_dir="plots"):
        self.function = function
        self.x_center = x_center
        self.plot_dir = plot_dir

        # Ensure the plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        self.hdc = HybridDerivativeCalculator(function, x_center)

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
        # Set the figure size
        plt.figure(figsize=(20, 10))
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
        nd_derivative = nd.Derivative(self.function)(self.x_center)

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
        plt.savefig(f"{self.plot_dir}/derivation_comparison_hist.png", dpi=300)

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
        y_values = [self.function(x) + np.random.randn(1)[0] * noise_std for x in x_values]

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
        plt.savefig(f"{self.plot_dir}/stem_demonstration.png", dpi=300)

        plt.show()

    def stem_method_with_fitting(self, x_values, y_values, min_samples=5):
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
        while max_spread >= 0.01 and len(x_values) >= min_samples:
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

        return float(slope), float(intercept), float(max_spread)

    def plot_derivatives_boxplot(self, iterations=100, noise_std=0.01):
        stem_derivatives = []
        stencil_derivatives = []

        for _ in range(iterations):
            def noisy_func(x, func=self.function):
                return func(x) + np.random.randn(1)[0] * noise_std

            # Recreate calculator with noisy function
            hdc_noisy = HybridDerivativeCalculator(noisy_func, self.x_center)

            # Calculate derivatives
            stem_derivatives.append(hdc_noisy.stem_method())
            stencil_derivatives.append(hdc_noisy.five_point_stencil_method())

        # Create two subplots for each method
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        stem_color = 'yellowgreen'
        stencil_color = 'hotpink'

        axes[0].boxplot(stencil_derivatives, labels=['Five-Point Stencil'], vert=False,
                        patch_artist=True, boxprops=dict(facecolor=stencil_color),
                        medianprops=dict(color='black'), meanline=True)

        axes[1].boxplot(stem_derivatives, labels=['Stem Method'], vert=False,
                        patch_artist=True, boxprops=dict(facecolor=stem_color),
                        medianprops=dict(color='black'), meanline=False)

        axes[1].set_xlabel('Derivative', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/derivation_comparison_boxplot.png", dpi=300)
        plt.show()
