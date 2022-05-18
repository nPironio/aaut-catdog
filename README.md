# aaut-catdog

## Setup
- Crear un virtual enviroment
- Instalar el paquete `catdog`:
    - moverse a `src/catdog`
    - correr `python setup.py develop --user`
- Instalar dependencias corriendo `./install_script.sh`.
 y extraer el contenido ahí

## Data
- Crear una carpeta `data`
- El dataset utilizado se puede descargar 
    - manualmente [acá](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection).
    - También se puede descargar usando la [kaggle CLI](https://github.com/Kaggle/kaggle-api) con el comando:
  
          kaggle datasets download -d andrewmvd/dog-and-cat-detection
- Extraer el contenido del zip descargado dentro de la carpeta `data`.
- Correr el comando `python src/preprocess_split.py`

## Codigo

### Preprocesamiento
correr el script `preprocess_split.py` para generar los archivos csv con la información de las instancias