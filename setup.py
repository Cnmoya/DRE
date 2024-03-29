import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DRE",
    version="0.0.1",
    author="Cristobal Moya, Felipe Urcelay",
    author_email="fjurcelay@uc.cl",
    description="completar descripcion...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cnmoya/DRE",
    project_urls={
        "Bug Tracker": "https://github.com/Cnmoya/DRE/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astrophysics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=['numpy', 'scipy', 'astropy', 'h5py', 'photutils', 'tqdm'],
    package_data={
        'DRE': ['models/*'],
    },
    scripts=['bin/dre', 'bin/make_cuts', 'bin/sex_dre', 'bin/psfex_dre'],
)
