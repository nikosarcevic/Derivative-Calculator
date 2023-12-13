# Derivative Calculator

The DerivativeCalculator class in this Python module is specifically tailored for accurately computing the derivative of complex functions at a designated point, utilizing the **stem method** explained in Stamera et al. 2016 *"SKA Weak Lensing III: Added Value of Multi-Wavelength Synergies for the Mitigation of Systematics"* ([see arXiv:1606.03451
](https://arxiv.org/abs/1606.03451)). It also features a five-point stencil method (used to showcase the superorority of the stem method), along with tools for demonstrating the method visually.

The motivation behind this code is rooted in addressing challenges in calculating derivatives within Fisher forecasting analysis (or any other occurance where the sability of numerical derivatives is central). Given the intricate nature of the functions involved, often accompanied by substantial noise, precise derivative computation becomes critical. The complexity of these functions, coupled with the difficulty in determining ideal derivative step sizes for various cosmological parameters, significantly impacts the stability of [Fisher matrices](https://en.wikipedia.org/wiki/Fisher_information). Traditional finite methods like the [five-point stencil](https://en.wikipedia.org/wiki/Five-point_stencil) or [numdifftools](https://numdifftools.readthedocs.io/en/master/) inadequate in such scenarios.

The stem method emerges as a robust alternative, particularly effective when the original function is complex and sensitive to minor parameter adjustments. This sensitivity typically leads to high variability in the derivatives with respect to each parameter, especially when small changes are evaluated. To address this, determining a bespoke and optimal step size for each parameter's derivative is essential. Alternatively, the stem method provides a practical solution to these challenges.

# The Power of Stem Method

The power of stem method lies in the fact that the derivative is obtained as follows:
 - first, the function is evaluated at a central value and several more values (arbitrary choice. We are following values given in Stamera et al.)
 - these values are collected in an array (or a list) and then a linear fit is performed
 - the derivative is then the slope of the linear function
A visual demonstration can be seen in figure:

![A showcase of the idea behind the stem method: function is evaluated multiple times, each value (purple scatter) is plotted and then linearly fitted (green line). The derivative is then the slope of the linear function.](/plots/stem_demonstration.png)

Derivative value evaluation using the stem method, compared to a finite difference method, such as five point stencil, is prefereed as the variance of the distribution of the derivatives (evaluted over some noisy data for example), will be extremely small. This can be seen in the histograms below:
![A range of derivatives estimated with a five point stencil method (hot pink) and stem method (yellow green) for a cubic function with introduced noise and evaluated 1000 times. The spread in the stem method case is extremely small compared to the finite difference method. The small variance is preferrable when the stability of numerical derivatives is required, such as Fisher analyses and similar. Value of the derivative obtained using the numdifftools library is also displayed in the subplots.](/plots/derivation_comparison_hist.png)

Alternative way of demonstrating the difference in variance of the distribution of the derivatives for a five point stencil and stem method is a box plot:
![Box plots for five point stencil (top) and stem methods (bottom) for a cubic function with gaussian noise.](/plots/derivation_comparison_boxplot.png)

> [!NOTE]
> Note that this implementation of the stem method routine is very simple and basic. Secondly, some choices (for example, the value of the deviations from the central value, the limit of the maximum spread, etc.) are arbitrary. It can be changed or improved depending on the purpose and the framework where it will be implemented.

# Features

**Stem Method**: Implements the stem derivative calculation based on the method developed by Camera et al.
This method is described in detail in *"SKA Weak Lensing III: Added Value of Multi-Wavelength Synergies for the Mitigation of Systematics"* ([see arXiv:1606.03451
](https://arxiv.org/abs/1606.03451)).

**Five-Point Stencil Method**: Provides a numerical differentiation using the five-point stencil method (for more details check [Wikipedia](https://en.wikipedia.org/wiki/Five-point_stencil), for example).

**Handling Output Noise**: Includes functionality for adding Gaussian noise to the function output and calculating derivatives with this noise.

**Visualization Tools**: Methods for plotting derivative distributions and demonstrating the stem method with noisy data.

# Installation

To use the DerivativeCalculator, you need to have Python installed along with the following libraries:

- NumPy
- Matplotlib
- Seaborn
- Numdifftools
- SciPy

You can install these dependencies using pip:

bash
```pip install numpy matplotlib seaborn numdifftools scipy```

# Usage

Here is a basic example of how to use the DerivativeCalculator:

python
```
import numpy as np
from derivative_calculator import DerivativeCalculator
```

Define your function
```
def my_function(x):
return np.sin(x)
```

Create an instance of the DerivativeCalculator
```
calc = DerivativeCalculator(myfunc=my_function, x_center=np.pi/4)
```

Calculate the derivative using the stem method
```
stem_derivative = calc.stem_method()
print("Stem Method Derivative:", stem_derivative)
```

Calculate the derivative using the five-point stencil method
```
stencil_derivative = calc.five_point_stencil_method()
print("Five-Point Stencil Derivative:", stencil_derivative)
```

# Contributing

Contributions to improve DerivativeCalculator are welcome. Please ensure to follow the code standards and add unit tests for any new features.

# License

MIT License

# Thanks

Thank you to Stefano Camera for the method, and Matthijs van der Wild and Marco Bonici for useful discussion.
