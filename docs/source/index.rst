Welcome to derivkit!
====================

Derivative calculator using STEM and stencil methods for noisy or discontinuous functions.

Install
-------

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ derivkit

Basic Usage
-----------

.. code-block:: python

   from derivkit.hybrid import STEMDerivatives
   result = STEMDerivatives(myfunc, x_center).calculate()

Notebooks
---------

For full usage, see the `example notebooks <https://github.com/nikosarcevic/derivkit/tree/main/notebooks>`_.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   derivkit
   notebooks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
