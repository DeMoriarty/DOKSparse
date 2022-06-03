from setuptools import setup, find_packages
from sparse_dok.__version__ import __version__
try:
  import cupy
except ImportError:
  print("Install cupy first.")
  exit()

setup(
  name = 'sparse_dok',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'sparse dok tensor implementation',
  author = 'demoriarty', 
  author_email = 'sahbanjan@gmail.com',
  url = 'https://github.com/DeMoriarty/SparseDOK',
  download_url = f'https://github.com/DeMoriarty/SparseDOK/archive/v_{__version__.replace(".", "")}.tar.gz',
  keywords = ['pytorch', "sparse"],
  install_requires=[ 
    'numpy',
    'torch>=1.10.0',
    'sympy'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
  include_package_data = True,
)