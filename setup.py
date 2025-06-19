from setuptools import find_packages, setup

__version__ = "0.1"


def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if not line.startswith("#")]


pkg_name = "torch_landscape"
requirements = parse_requirements("requirements.txt")
setup(
    name=pkg_name,
    version=__version__,
    install_requires=requirements,
    packages=find_packages(where=pkg_name),
)
