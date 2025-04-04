from setuptools import Extension, setup
from Cython.Build import cythonize
import os
import platform

# Automatically collect all C++ source files
def get_cpp_sources():
    sources = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".cpp") and file != "PyPTRouter.cpp":
                sources.append(os.path.join(root, file))
    return sources

# Create a simple extension module
ext = Extension(
    name="PyPTRouter",  # Module name for import
    sources=[
        "PyPTRouter.pyx",  # Cython source file
        *get_cpp_sources()  # All C++ source files
    ],
    include_dirs=["."],  # Directory containing header files
    language="c++",
    extra_compile_args=["-std=c++20"] if platform.system() != "Windows" else ["/std:c++20"],
)

# Configure the setup
setup(
    ext_modules=cythonize(
        ext,
        compiler_directives={
            "language_level": 3,
            "embedsignature": True,
        },
    ),
) 