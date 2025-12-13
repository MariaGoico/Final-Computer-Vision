# Final-Computer-Vision

Pasos Trabajo Final:
1. Dataset de fotos de sitios (aprox 1000)
	- Hacer cutout y predecir con el denoise
	- Metric: mse (?)
	- U-Net -> modificar los segmentations models
    También debería servir para eliminar otros artefactos de la imagen.
2. Coger la mascara y restarle a la imagen de la valla
	- Inferir -> Va o no va?
Si funciona:
	- Segmentador de vallas -> detectar, poner a 0 y rellenar
Si no: se verá.