# aaut-catdog

## Setup
- Crear un virtual enviroment
- Instalar dependencias corriendo `./install_script.sh` desde el directorio root.


## Data
- Crear una carpeta `data`
- El dataset utilizado se puede descargar 
    - manualmente [acá](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection).
    - También se puede descargar usando la [kaggle CLI](https://github.com/Kaggle/kaggle-api) con el comando:
  
          kaggle datasets download -d andrewmvd/dog-and-cat-detection
- Extraer el contenido del zip descargado dentro de la carpeta `data`.
- Correr el comando `python src/preprocess_split.py`
