from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Align image tiles acquired with SBEM into an image stack. Using SOFIMA (by Google Research)'
LONG_DESCRIPTION = ''

# Setting up
setup(
        name='emalign', 
        version=VERSION,
        author='Valentin Gillet',
        author_email='valentin.gillet@biol.lu.se',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy>=1.26,<2.3',
            'pandas>=2.2',
            'networkx>=3.2',
            'opencv-python>=4.10,<5',
            'tensorstore>=0.1.60',
            'scipy>=1.12',
            'pymongo>=4.6',
            'tqdm>=4.66',
            'jax>=0.4.30,<0.6',
            'jaxlib>=0.4.30,<0.6',
            'sofima @ git+https://github.com/google-research/sofima.git@5c5642cded4b33ec7e975afc3d85f15c0126fb2e'
        ],
        extras_require={
            'neuroglancer': ['neuroglancer>=2.41'],
            'cuda12': [
                'jax[cuda12]>=0.4.30,<0.6; platform_system=="Linux"'
            ]
        },
        keywords=['python', 'alignment']
    )
