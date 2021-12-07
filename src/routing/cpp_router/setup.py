import setuptools  # important
import numpy
from distutils.core import Extension, setup
from Cython.Build import cythonize

""" to install an use the cpp-routers the c++ file have to be compiled and
the interface to python via cython has to be created
therefore a c++ compiler has to be installed and linked correctly to python
    (dont ask me how. on windows it works after installing visual studio code 2019 with the python extension)
then run:
    python setup.py build_ext --inplace
this has to be redone when changes in the c++-files have been made

the setup should create a .pyd-file and a build-folder
"""

ext = Extension(name="PyNetwork", sources=["PyNetwork.pyx"], build_dir="build",
                                           script_args=['build'], 
                                           options={'build':{'build_lib':'.'}},
                                            include_dirs=[numpy.get_include()])

setup(ext_modules=cythonize(ext))