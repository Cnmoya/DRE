# DRE: Discrete Radius Ellipticals

Contenidos
==========

<!--ts-->
* [About]
* [Version Notes]
* [Requirements]
* [Installation]
* [How to use DRE]
  * [Preprocessing]
  * [Executing DRE]
    * [Locally (CPU)]
    * [In Colab (GPU)]
* [Working directory]
* [Scripts]
  * [sex_dre]
  * [psfex_dre]
  * [make_cuts]
  * [dre]
<!--te-->

## About
A python program to fit elliptical galaxies (De Vaucouleurs's profile) in astronomical images, 
it seeks to be a fast and efficient program using a set of precalculated models and CPU/GPU acceleration.

## Version Notes
* Option to use cubes with variable Sersic index
* Fits a parabola to the residuals to find a minimum by interpolation
* Visualize residuals cube in Colab
* Change the summary format to .fits

## Requirements
DRE requieres the following python packages that will be installed automatically when installing DRE with PIP:
* Numpy
* Scipy
* Astropy
* Photutils
* H5Py

For running DRE with GPU it also requires:
* Cupy
But a local installation is not needed if running on Google Colab.

For the preprocessing and to obtain the PSF:
* SExtractor
* PSFex

## Installation

First clone this repo:
```
git clone https://github.com/Cnmoya/DRE.git
```
You can make any modification at this point. To install it:
```
pip install --use-feature=in-tree-build ./DRE
```

Note: The models files in `DRE/DRE/models` will not be downloaded with `git clone` at least you have installed
[git-lfs](https://git-lfs.github.com/) for large files support, if you can't install git-lfs
you can download de repo as zip file.

# How to use DRE

The general procedure for running DRE is:
`Images -> Sextractor -> PSFex -> Cuts -> DRE`

### Preprocessing
DRE requires the results of [SExtractor](https://sextractor.readthedocs.io/en/stable/) to extract the sources and 
separate stars from galaxies, in particular requires the check images `-BACKGROUND`, `BACKGROUND_RMS` y `SEGMENTATION`, 
and the `X_IMAGE`, `Y_IMAGE` and `CLASS_STAR` parameters, the catalogs should be in fits format.

It also requires a PSF for the input image, this can be extracted with
[PSFex](https://psfex.readthedocs.io/en/latest/GettingStarted.html), is important to set `PSF_STEP = 1` in the
configuration so that its resolution is the same as the image. DRE will use the PSF to order 0, 
so it is not necessary to calculate it to a higher order.

To facilitate this procedure, DRE includes the scripts `sex_dre` y `psfex_dre` (more information in the [scripts] section).

## Cuts
Once you have the SExtractor catalogs, you must cut out the extracted objects, the segments and the noise.
For this you can use the script `make_cuts` which generates cutouts of 128x128 pixels centered on the object and saves them to an HDF5 file.

## Executing DRE
Finally, to run DRE you can do it in two ways:

### Locally (CPU)
You can run DRE in parallel with the `dre` script, this option is convenient if you want to run it for a long time on a very long list of files.

### In Colab (GPU)
DRE can also use GPU acceleration with CUDA, this option allows calculations at a higher speed than CPU. 
DRE can be used with Google Colab platform, which allows you to run Python interactively on a Google server. 
This option is convenient to perform calculations quickly and analyze the results on the same platform, 
but the downside is that the Colab times for GPUs are limited.

As an example of how to use DRE in Colab you can see the following notebook:
<a href="https://colab.research.google.com/github/Cnmoya/DRE/blob/master/Example/DRE_Example.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Working directory
For automation DRE performs calculations on all files in a directory, below there is an example of the structure 
of the working directory that is obtained when executing DRE with the default parameters, where `Tiles` contains the science images, `Sextracted` the 
results of executing `sex_dre`, `PSF` the results of `psfex_dre`, `Cuts` of `make_cuts` and `Chi`, `Summary` and `Mosaics` the results of `dre`.

All the resulting files for a science image will have se same name but with a different suffix `_xxx` or extension.
```markdown
.
├── Tiles
│   ├── image1.fits
│   └── image2.fits
├── Sextracted
│   ├── image1
│   │   ├── image1_cat.fits   # SExtractor catalog
│   │   ├── image1_nb.fits    # -BACKGROUND
│   │   ├── image1_rms.fits   # BACKGROUND_RMS
│   │   └── image1_seg.fits   # SEGMENTATION
│   └── image2
│       ├── image2_cat.fits
│       ├── image2_nb.fits
│       ├── image2_rms.fits
│       └── image2_seg.fits
├── PSF
│   ├── image1.psf
│   └── image2.psf
├── Cuts
│   ├── image1_cuts.h5
│   └── image2_cuts.h5
├── Chi
│   ├── image1_chi.h5
│   └── image2_chi.h5
├── Summary
│   ├── image1.dat
│   └── image2.dat
└── Mosaics
    ├── image1
    │   ├── image1_00_0001_mosaic.fits   
    │   ├── image1_00_0002_mosaic.fits
    │   └── ...
    └── image2
        ├── image2_00_0001_mosaic.fits
        ├── image2_00_0002_mosaic.fits
        └── ...
```

## Scripts
DRE includes the following scripts:
*  `sex_dre`
*  `psfex_dre`
*  `make_cuts`
*  `dre`

Will be added to the `/bin` directory of the python environment, so you can use them from the terminal. You can use the `-h`/`--help` parameter to display all options. 

### sex_dre
Runs SExtractor on all files in the `Tiles` directory with the same configuration.
```
$ sex_dre --help
usage: sex_dre [-h] [-i INPUT] [-o OUTPUT] [-c CONFIG] [--subdir] [--flags FLAGS] [--weights WEIGHTS] [--gain GAIN] [--gain-key GAIN_KEY]

Wrapper for running SExtractor with DRE parameters

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        directory with input images to be SExctracted
  -o OUTPUT, --output OUTPUT
                        output directory for sextracted data (def: Sextracted)
  -c CONFIG, --config CONFIG
                        SExtractor configuration file (def: default.sex)
  --subdir              make a subdirectory for each image
  --flags FLAGS         directory with fits images to be passed as FLAG_IMAGE, must have the same name as the input image but ending in '_flag'
  --weights WEIGHTS     directory with fits images to be passed as WEIGHT_IMAGE, must have the same name as the input image but ending in '_wht'
  --gain GAIN           Can be 'counts' to read from the header and it as GAIN, 'cps' to read from the header and use GAIN*EXPOSURE_TIME, or a float in e-/ADU to be used directly as GAIN
  --gain-key GAIN_KEY   Header key for gain if 'counts' or 'cps' are used or 'auto' to search in the header (Def: auto)
```
If it is required to pass additional images such as Flags or Weight Maps, 
it can be done with the corresponding parameters using the same structure for those directories. 
It also allows to calculate the gain of each image from the header values.

For example to run `sex_dre` on` Tiles` calculating the gain for images in counts per second (`gain * exposure_time`),
Using the configuration file `sex_source/default.sex`, from the root of the working directory execute:
```
$ sex_dre -c sex_source/default.sex --gain cps
```

### psfex_dre
Extracts a PSF for every SExtractor catalog.
```
$  psfex_dre --help
usage: psfex_dre [-h] [-i INPUT] [-o OUTPUT] [-c CONFIG]

Wrapper for running PSFex with DRE parameters

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        directory with SExtractor catalogs (def: Sextracted)
  -o OUTPUT, --output OUTPUT
                        output directory for the DRE PSF's (def: PSF)
  -c CONFIG, --config CONFIG
                        PSFex configuration file (def: default.psfex)

```

### make_cuts
It makes cuts from SExtractor catalogs, you can control the stelarity parameter to exclude stars, 
a margin to discard objects at the edge of the image, and the degree of compression of the output HDF5 files.

```
$ make_cuts --help
usage: make_cuts [-h] [-i INPUT] [-s SEXTRACTED] [-o OUTPUT] [--margin MARGIN] [--max-stellarity MAX_STELLARITY] [--compression {none,low,medium,high}] [--warn]

Cut images of 128x128 centered in SExtracted objects

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        root directory to original data (def: Tiles)
  -s SEXTRACTED, --sextracted SEXTRACTED
                        root directory to SExtracted data (def: Sextracted)
  -o OUTPUT, --output OUTPUT
                        output directory (def: Cuts)
  --margin MARGIN       margin to exclude objects near the borders of the image (def: 80)
  --max-stellarity MAX_STELLARITY
                        maximum SExtractor 'CLASS_STAR' stelarity to be considered a galaxy (def: 0.5)
  --compression {none,low,medium,high}
                        compresion level for the h5 output file,lower is faster (def: medium)
  --warn                Print warnings

```

### dre
Runs the DRE fit over each object cut, for each image it performs a convolution of the model with the corresponding PSF before the fit. 
It also allows executing it in parallel with the `--cpu` argument and controlling the degree of compression of the output HDF5 files.

```
$ dre --help
usage: dre [-h] [--psf PSF] [-i INPUT] [-o OUTPUT] [--mosaics] [--cpu CPU] [--chunk CHUNK] [--compression {none,low,medium,high}] [--warn] model

Run DRE with multiprocessing

positional arguments:
  model                 models cube file

optional arguments:
  -h, --help            show this help message and exit
  --psf PSF             directory with PSF's (def: PSF)
  -i INPUT, --input INPUT
                        directory with cuts (def: Cuts)
  -o OUTPUT, --output OUTPUT
                        output directory (def: Chi)
  --mosaics             generate mosaics files for visualization
  --cpu CPU             Number of cpu's to use
  --chunk CHUNK         Max size of the queue
  --compression {none,low,medium,high}
                        compresion level for the h5 output file,lower is faster (def: medium)
  --warn                Print warnings

```

For example, to run DRE with 4 CPU's and generate mosaics to visualize the resulting model and
the residual:
```
$ dre --cpu 4 --mosaics
```

Note: The installation will not include the models unless you have installed [git-lfs](https://git-lfs.github.com/)
so if you don't add them manually before installation you have to pass the models file as an argument, for example:
```
$ dre -m modelbulge.fits
```

[About]: #about
[Version Notes]: #version-notes
[Requirements]: #requirements
[Installation]: #installation
[How to use DRE]: #how-to-use-dre
[Preprocessing]: #preprocessing
[Executing DRE]: #executing-dre
[Locally (CPU)]: #locally-cpu
[In Colab (GPU)]: #in-colab-gpu
[Working directory]: #working-directory
[Scripts]: #scripts
[sex_dre]: #sex_dre
[psfex_dre]: #psfex_dre
[make_cuts]: #make_cuts
[dre]: #dre
