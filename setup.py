#!/usr/bin/env python

"""
setup.py  to build rst code with cython
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy # to get includes

extensions = [Extension("rst", ["src/rst.pyx"])]

setup(
    name='rst',
    version='0.1',
    description='Random Spanning Trees',
    author='André Panisson',
    author_email='panisson@gmail.com',
    url='andre.panisson.com',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions,
                            compiler_directives={'language_level' : "3"}),
    include_dirs = [numpy.get_include(),],
)