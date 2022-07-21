import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.9'
PACKAGE_NAME = 'Data42'
AUTHOR = 'Pablo Diaz Gonzalez'
AUTHOR_EMAIL = 'pablo.diaz.92@outlook.com'
URL = 'https://github.com/pablod1/library-project-TheBridge'

LICENSE = 'MIT'
DESCRIPTION = 'Library specializing in data cleaning, visualization and prediction'
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"


INSTALL_REQUIRES = [
    'pandas',
    'numpy',
    'seaborn',
    'matplotlib',
    'seaborn',
    'sklearn',
    'scipy',
    'imblearn'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)