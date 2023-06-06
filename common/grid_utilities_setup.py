# import setuptools  # important
from distutils.core import setup, Extension
import numpy as np

setup(name = 'grid_utilities',
      version = '1.0',
      ext_modules = [Extension('grid_utilities', ['grid_utilities.c'],
                     include_dirs=[np.get_include()],
                     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])])