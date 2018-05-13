from setuptools import setup, find_packages

install_requires = ['menpo3d>=0.6.0']

setup(name='itwmm',
      version='1.0.0',
      description='ITW Morphable Model fitting',
      author='Menpo Authors',
      author_email='hello@menpo.org',
      packages=find_packages(),
      install_requires=install_requires
      )
