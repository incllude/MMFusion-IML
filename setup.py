"""
Setup script for MMFusion-IML: Multi-Modal Fusion for Image Manipulation Detection and Localization
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mmfusion_iml",
    version="0.1.0",
    author="Konstantinos Triaridis, Vasileios Mezaris",
    author_email="",
    description="Multi-Modal Fusion for Image Manipulation Detection and Localization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/MMFusion-IML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "mmfusion_iml": [
            "configs/*.py",
            "experiments/*.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmfusion-train=mmfusion_iml.scripts.train:main",
            "mmfusion-test=mmfusion_iml.scripts.test:main",
            "mmfusion-inference=mmfusion_iml.scripts.inference:main",
        ],
    },
)
