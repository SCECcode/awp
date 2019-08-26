""" Installs OpenFD using distutils

    Supports the additional commands
    >> python setup.py clean 

"""
import sys
import subprocess
import os
import shutil

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command


root_dir = os.path.dirname(os.path.realpath(__file__))

class clean(Command):
    """ Remove build dirs, and .pyc files
    """
    
    description = "Remove build dirs and files"
    user_options = []

    def run(self):

        # Remove .pyc-files
        curr_dir = os.getcwd()
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pyc') and os.path.isfile:
                    os.remove(os.path.join(root, file))

        # Remove build directories
        os.chdir(root_dir)
        names = ["MANIFEST", "build", "dist"]

        for f in names:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass



if __name__ == '__main__':
    setup(
        name="OpenFD",
    
        version="0.1.0",
    
        author="OpenFD development team",
        author_email="ossian.oreilly@gmail.com",
    
        packages=["openfd"],
    
        include_package_data=True,
    
        url="http://pypi.python.org/pypi/tbd/",
    
        cmdclass={
                  'clean': clean,
                 },
    
        license="LICENSE.txt",
        description="",
    
        # Dependent packages (distributions)
        install_requires=[
            "sympy","numpy", 'pyopencl',
        ],
        )

