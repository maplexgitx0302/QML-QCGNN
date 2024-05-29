from setuptools import setup, find_packages

setup(
    name='QCGNN',
    version='1.0.0',
    description='Jet Discrimination with Quantum Complete Graph Neural Network',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yi-An Chen',
    author_email='maplexworkx0302@gmail.com',
    url='https://github.com/maplexgitx0302/QML-QCGNN',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # Common packages,
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'seaborn',
        
        # HEP related
        'awkward',
        'gdown',
        'h5py',
        'uproot',
        'tables',

        # Machine Learning
        'pennylane',
        'pennylane-qiskit',
        'torch',
        'torch_geometric',

        # ML Tools
        'lightning',
        'wandb',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Physics :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
