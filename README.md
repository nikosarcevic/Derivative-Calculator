# Derivative Calculator

This Python module provides a DerivativeCalculator class, designed for calculating the derivative of a given function at a specified point using different methods. It includes implementations of the stem method and the five-point stencil method, as well as capabilities for handling output noise and visualizing the results.

# Features

Stem Method: Implements the stem derivative calculation based on the method developed by Camera et al.
Five-Point Stencil Method: Provides a numerical differentiation using the five-point stencil method.
Handling Output Noise: Includes functionality for adding Gaussian noise to the function output and calculating derivatives with this noise.
Visualization Tools: Methods for plotting derivative distributions and demonstrating the stem method with noisy data.
Installation

To use the DerivativeCalculator, you need to have Python installed along with the following libraries:

NumPy
Matplotlib
Seaborn
Numdifftools
SciPy
You can install these dependencies using pip:

bash
Copy code
pip install numpy matplotlib seaborn numdifftools scipy
Usage

Here is a basic example of how to use the DerivativeCalculator:

python
Copy code
import numpy as np
from derivative_calculator import DerivativeCalculator

## Define your function
def my_function(x):
    return np.sin(x)

## Create an instance of the DerivativeCalculator
calc = DerivativeCalculator(myfunc=my_function, x_center=np.pi/4)

## Calculate the derivative using the stem method
stem_derivative = calc.stem_method()
print("Stem Method Derivative:", stem_derivative)

## Calculate the derivative using the five-point stencil method
stencil_derivative = calc.five_point_stencil_method()
print("Five-Point Stencil Derivative:", stencil_derivative)
Contributing

Contributions to improve DerivativeCalculator are welcome. Please ensure to follow the code standards and add unit tests for any new features.

# License

MIT License
