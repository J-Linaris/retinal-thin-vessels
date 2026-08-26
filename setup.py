"""
File with the only purpose of compiling the
'dse_helper' cython file
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="retinal_thin_vessels.external.DSE_skeleton_pruning.dsepruning.dse_helper",
        
        sources=["retinal_thin_vessels/external/DSE_skeleton_pruning/dsepruning/dse_helper.pyx"], 
        
        include_dirs=[numpy.get_include()],

        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    ext_modules=cythonize(extensions)
)