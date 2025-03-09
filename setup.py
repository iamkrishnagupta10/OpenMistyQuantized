from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="openmisty",
    version="0.1.0",
    description="OpenMisty: A quantized and optimized version of OpenManus for running AGI locally",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Krishna Gupta",
    author_email="iamkrishnagupta10@gmail.com",
    url="https://github.com/iamkrishnagupta10/OpenMistyQuantized",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "openmisty=app.main:main_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["prompt/*.txt", "prompt/*.md"],
    },
) 