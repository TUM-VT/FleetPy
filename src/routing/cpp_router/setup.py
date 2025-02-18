import setuptools  # important
import numpy
from distutils.core import Extension, setup
from Cython.Build import cythonize

""" to install an use the cpp-routers the c++ file have to be compiled and
the interface to python via cython has to be created
therefore a c++ compiler has to be installed and linked correctly to python
    (Dont ask me how. on windows it works after installing visual studio code 2019 with the python extension. Other useful resource for Windows you could try is https://stackoverflow.com/a/56811119. I also installed this compiler: https://www.freecodecamp.org/news/how-to-install-c-and-cpp-compiler-on-windows/ (but I think was already installed in the previous step too, so you can try to skip this for now and, if the process fails, try with this compiler)
then run:
    python setup.py build_ext --inplace
this has to be redone when changes in the c++-files have been made

If "python setup.py build_ext --inplace" leads to problems, try:
1) Open the special command prompt -> https://stackoverflow.com/a/41724634
2) Activate the environment you use with fleetpy (or in which you have installed cython)
3) python setup.py build_ext --inplace

the setup should create a .pyd-file and a build-folder
"""

ext = Extension(name="PyNetwork", sources=["PyNetwork.pyx"], build_dir="build",
                                           script_args=['build'], 
                                           options={'build':{'build_lib':'.'}},
                                            include_dirs=[numpy.get_include()])

setup(ext_modules=cythonize(ext))
