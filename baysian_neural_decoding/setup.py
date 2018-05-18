from setuptools import setup

setup(name='baysian_neural_decoding',
      version='1.0',
      description='Decoding method from Insanally, et al. 2015',
      long_description='',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis'],
      url='',
      author='Badr F. Albanna',
      author_email='badr.albanna@gmail.com',
      license='MIT',
      packages=['baysian_neural_decoding'],
      install_requires=[
        'h5py',
        'numpy',
        'scipy'
        'statsmodels'],
      zip_safe=False)