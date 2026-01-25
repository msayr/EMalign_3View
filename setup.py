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
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'opencv-python>=4.5.0',
            'tensorstore>=0.1.45',
            'jax>=0.4.0',
            'sofima',
            'connectomics',
            'emprocess',
            'networkx>=2.6.0',
            'tqdm>=4.60.0',
        ],
        extras_require={
            'mongo': ['pymongo>=4.0.0'],
            'viz': ['neuroglancer>=2.0.0'],
            'all': [
                'pymongo>=4.0.0',
                'neuroglancer>=2.0.0',
            ],
        },
        keywords=['python', 'alignment']
    )
