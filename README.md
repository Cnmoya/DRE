# DRE
Discrete Radius Ellipticals

## Notas de la versión
* DRE es instalable via pip (localmente).
* Se incluyen scipts para SExtractor, los cortes y correr DRE con cpu.
* Se puede correr de forma interactiva con Colab.  
* Corrección en la forma de calcular chi cuadrado.

## Instalación

Primero clona esta repo:
```
git clone https://github.com/Cnmoya/DRE.git
```
Puedes hacer las modificaciones que quieras antes de instalarlo. Para instalar desde la copia local:
```
pip install ./DRE
```

## Como usar DRE
DRE requiere los resultados de [SExtractor](https://sextractor.readthedocs.io/en/stable/) para extraer las fuentes y 
separar estrellas de galaxias, en particular requiere las imágenes de chequeo `-BACKGROUND`, `BACKGROUND_RMS` y `SEGMENTATION`, 
y los parámetros `X_IMAGE`, `Y_IMAGE` y `CLASS_STAR`. Además requiere una PSF para la imagen de entrada, esta puede ser extraída con 
[PSFex](https://psfex.readthedocs.io/en/latest/GettingStarted.html).

```markdown
.
├── Tiles
│   ├── image1.fits
│   └── image2.fits
├── sex_source
│   ├── default.nnw
│   ├── default.param
│   ├── default.sex
│   └── default.psfex
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
│   ├── image1_tab.fits
│   └── image2_tab.fits
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