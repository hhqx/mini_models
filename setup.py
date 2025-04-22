from setuptools import setup, find_packages
import os
import re

def get_version():
    """获取版本号从 mini_models/version.py 文件"""
    version_file = os.path.join("mini_models", "version.py")
    with open(version_file, "r") as f:
        content = f.read()
    match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
    if match:
        return match.group(1)
    raise RuntimeError(f"Unable to find version string in {version_file}")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mini_models",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight deep learning model library for resource-constrained environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hhqx/mini_models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
)
