# aaut-catdog

## Setup
- Crear un virtual enviroment
- Instalar el paquete `catdog`:
    - moverse a `src/catdog`
    - correr `python setup.py develop --user`
- Instalar los requirements en `requirements.txt`
- Descargar la data de https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection
- Crear una carpeta `data` y extraer el contenido ahí

## Codigo

### Preprocesamiento
correr el script `preprocess_split.py` para generar los archivos csv con la información de las instancias