from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GridDataGen",
    version="1.0",
    author="Matteo Mazzonelli, Alban Puech, Jonas Weiss",
    author_email="Matteo.Mazzonelli@ibm.com , Alban.Puech1@ibm.com, jwe@zurich.ibm.com",
    description="Grid Data Generation",
    packages=find_packages(),
    install_requires=requirements,
)
