# DRE
Discrete Radius Ellipticals

Contenidos
==========

<!--ts-->
* [Acerca de]
* [Notas de la version]
* [Requerimientos]
* [Instalacion]
* [Como usar DRE]
  * [Preprocesamiento]
  * [Recotes]
  * [Ejecutar DRE]
    * [En CPU]
    * [En GPU]
* [Directorio de trabajo]
* [Scripts]
  * [sex_dre]
  * [psfex_dre]
  * [make_cuts]
  * [dre]
<!--te-->

## Acerca de
descripción...

## Notas de la version
* DRE es instalable via pip (localmente).
* Se incluyen scipts para SExtractor, los cortes y correr DRE con cpu.
* Se puede correr de forma interactiva con Colab.  
* Corrección en la forma de calcular chi cuadrado.

## Requerimientos
DRE requiere los siguientes paquetes de python que serán instalados automáticamente al instalar DRE con PIP:
* Numpy
* Scipy
* Astropy
* Photutils
* H5Py

Para correr DRE con GPU se requieren adicionalmente:
* Cupy
* opt-einsum
Pero no es necesario instalarlos si usas DRE a través de Google Colab.

Para el preprocesamiento y para obtener la PSF se requieren:
* SExtractor
* PSFex

## Instalacion

Primero clona esta repo:
```
git clone https://github.com/Cnmoya/DRE.git
```
Puedes hacer las modificaciones que quieras antes de instalarlo. Para instalar desde la copia local:
```
pip install ./DRE
```

Nota: Los modelos que se encuentran en `/DRE/models` no se descargarán a menos que tengas instalado
[git-lfs](https://git-lfs.github.com/) para el soporte de archivos pesados.

# Como usar DRE

El procedimiento general para ejecutar DRE es:
`Images -> Sextractor -> PSFex -> Cuts -> DRE`

### Preprocesamiento
DRE requiere los resultados de [SExtractor](https://sextractor.readthedocs.io/en/stable/) para extraer las fuentes y 
separar estrellas de galaxias, en particular requiere las imágenes de chequeo `-BACKGROUND`, `BACKGROUND_RMS` y `SEGMENTATION`, 
y los parámetros `X_IMAGE`, `Y_IMAGE` y `CLASS_STAR`.

Además requiere una PSF para la imagen de entrada, esta puede ser extraída con
[PSFex](https://psfex.readthedocs.io/en/latest/GettingStarted.html), es importante que esta tenga un valor de `PSF_STEP = 1` para que su
resolución sea la misma de la imagen. DRE usará la PSF a orden 0, por lo que no es necesario calcularla a un orden mayor.

Para facilitar este procedimiento DRE incluye los scripts `sex_dre` y `psfex_dre` (más información en la sección de scripts).

## Recortes
Una vez se tienen los catálogos de SExtractor se deben realizar recortes de los objetos extraídos, los segmentos y el ruido.
Para esto se puede usar el script `make_cuts` que genera recortes de $128x128$ pixeles centrados en el objeto y los guarda en un archivo HDF5.

## Ejecutar DRE
Finalmente para ejecutar DRE puedes hacerlo de dos formas:

### En CPU
Puedes ejecutar DRE en paralelo con el script `dre`, esta opción es conveniente si deseas ejecutarlo de forma prolongada sobre una lista muy extensa de archivos

### En GPU
DRE también puede usar aceleración por GPU con CUDA, esta opción permite cálculos a una mayor velocidad que por CPU. La mejor forma de hacerlo es usando
la plataforma Google Colab, que permite ejecutar Python de forma interactiva en un servidor de Google. Esta opción es conveniente para realizar cálculos rápidamente
y analizar los resultados en la misma plataforma, pero el inconveniente es que en la versión gratuita de Colab los tiempos para GPU son limitados.

Como ejemplo de como usar DRE en Colab puedes ver nuestro [![Example Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

## Directorio de trabajo
Para facilitar la automatización DRE realiza los cálculos sobre todos los archivos en un directorio, a continuación se muestra un ejemplo de la estructura
del directorio de trabajo que se obtiene al ejecutar DRE con los parámetros por defecto, donde `Tiles` contiene las imágenes de ciencia, `Sextracted` los resultados
de ejecutar `sex_dre`, `PSF` los resultados de ejecutar `psfex_dre`, `Cuts` los recortes de `make_cuts` y `Chi`, `Summary` y `Mosaics` los resultados de `dre`.

Los archivos correspondientes a una imagen de ciencia tendrán el mismo nombre pero cambiando el sufjo `_xxx` o la extensión.
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

## Scripts
La instalación de DRE incluye los siguientes scripts:
*  `sex_dre`
*  `psfex_dre`
*  `make_cuts`
*  `dre`

Serán añadidos al directorio `/bin` de tu environment en la instalación por lo que podrás usarlos directamente desde el terminal.
Todos ellos incluyen una opción `-h`/`--help` con una descripción de todas sus opciones. Estos scipts están diseñados para usarse directamente si usas la
estructura de trabajo presentada anteriormente, pero puedes cambiar los nombres de los directorios con los argumentos si lo deseas.

### sex_dre
Permite ejecutar SExtractor en todo el directorio `Tiles` de forma secuencial usando un mismo archivo de configuración y generando los archivos en el formato  
adecuado para DRE.
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
Si se requiere pasar imágenes adicionales como Flags o Weight Maps puede hacerse con los parámetros correspondientes usando la misma estructura para esos directorios.
Además permite calcular la ganancia de cada imagen a partir de los valores del header.

Por ejemplo para ejecutar `sex_dre` en `Tiles` calculando la ganancia para imágenes en cuentas por segundo (`gain * exposure_time`),
usando el archivo de configuración `sex_source/default.sex`, desde la raíz del directorio de trabajo ejecuta:
```
$ sex_dre -c sex_source/default.sex --gain cps
```

### psfex_dre
Funciona de manera similar a `sex_dre`
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
Realiza los recortes a partir de los catálogos de SExtractor, se puede controlar el parámetro
de estelaridad para excluir estrellas, un margen para descartar los objetos en el borde de la imagen
y el grado de compresión de los archivos HDF5 de salida.

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
Ejecuta el ajuste de DRE sobre los recortes, para cada imagen realiza una convolución del modelo con la PSF
correspondiente antes del ajuste. Permite además ejecutarlo en paralelo con el argumento `--cpu`
y controlar el grado de compresión de los archivos HDF5 de salida.

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

La instalación (por ahora) no incluirá los modelos a menos que tengas instalado [git-lfs](https://git-lfs.github.com/)
por lo que un argumento obligatorio es el archivo con los modelos que debe pasarse en primer lugar.
Suponiendo que tienes los modelos en la raíz del directorio de trabajo podemos ejecutar DRE con 4 cpu's
con el siguiente comando:
```
$ dre modelbulge.fits --cpu 4
```

[Acerca de]: #acerca-de
[Notas de la version]: #notas-de-la-version
[Instalacion]: #instalacion
[Como usar DRE]: #como-usar-dre
[Requerimientos]: #requerimientos
[Preprocesamiento]: #preprocesamiento
[Recotes]: #recortes
[Ejecutar DRE]: #ejecutar-dre
[En CPU]: #en-cpu
[En GPU]: #en-gpu
[Directorio de trabajo]: #directorio-de-trabajo
[Scripts]: #scripts
[sex_dre]: #sex_dre
[psfex_dre]: #psfex_dre
[make_cuts]: #make_cuts
[dre]: #dre
