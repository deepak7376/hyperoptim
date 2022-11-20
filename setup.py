from setuptools import setup, find_packages
  
# reading long description from file 
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

  
# some more details 
CLASSIFIERS = [ 
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Developers', 
    'Topic :: Scientific/Engineering :: Mathematics', 
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3.0', 
    ] 
  
# calling the setup function  
setup(name='hyperoptim', 
      version='0.0.1', 
      description='Hyperparameter Optimization using Genetic Algorithms.', 
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/deepak7376/hyperoptim', 
      author='Deepak Yadav', 
      author_email='dky.united@gmail.com', 
      license='MIT', 
      py_modules=["hyperoptim"],
      package_dir={'':'src'},
      classifiers=CLASSIFIERS,  
      keywords='Hyperparameter optimization using Genetic Algorithms.',
      python_requires='>=3'

      ) 
