from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
        ext_modules = cythonize("ddm_data_simulation1.pyx", annotate=True),
        include_dirs = [numpy.get_include()]
    )
