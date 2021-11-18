from setuptools import setup, find_packages

requirements_txt = open("requirements.txt").read().split("\n")
requirements = list(filter(lambda x: "--extra" not in x and x is not "", requirements_txt))

setup(
    name="hijax",
    version="0.1.0",
    author="Angus Turner",
    author_email="angusturner27@gmail.com",
    url="https://github.com/angusturner/hijax",
    description="ML experiment framework for Jax and Haiku",
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
)
