# DRE
Discrete Radius Ellipticals

## Notas de la versión
* filtro de nan's en los recortes
* Corre en CPU con shared memory
* Corrección en la forma de calcular el ruido
* Cambia el orden de las dimensiones del cubo `(10, 13, 128, 21, 128) -> (10, 13, 21, 128, 128)`
* No se hace tiling a las imagenes
