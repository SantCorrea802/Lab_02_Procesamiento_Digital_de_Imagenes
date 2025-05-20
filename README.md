# Laboratorio 2 - Procesamiento Digital de Imágenes 2025-1 UdeA

## Set de datos utilizado
[Brain Tumor Dataset en Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

## Descripción
Este proyecto utiliza el set de datos *Brain Tumor Dataset* para detectar cuando una radiografía de cerebro presenta un tumor (clasificado como `1`) o cuando no presenta tumor (clasificado como `0`).

## NOTA
No es necesario descargar el dataset desde Kaggle, el repositorio ya cuenta con el set de datos de test_brain_tumor y test_healthy para poder probar el programa (archivo .py) tampoco se deben ejecutar los .ipynb ya que estos se ejecutan con respecto al dataset original de kaggle para poder realizar el EDA y entrenamiento de los modelos. Si desea ejecutar los nootbooks asegurese de tener la siguiente configuracion del set de datos:
```
Carpeta_principal/
├── Brain_Tumor_Data_Set/
│ ├── Brain_Tumor_Data_Set/
│ │ ├── Brain_Tumor/ # Imagenes de pacientes con tumor cerebral para entrenamiento y testeo
│ │ └── Healthy/ # Imagenes de pacientes sanos para entrenamiento y testeo
│ ├── metadata.csv # Metadatos generales del dataset
│ └── metadata_rgb_only.csv # Metadatos con imágenes RGB solamente
│
├── Model/
│ ├── entrenar_modelo.ipynb # Notebook para entrenar el modelo
│ ├── preparar_datos.ipynb # Notebook para preparar y procesar los datos
│ ├── preparar_modelos.ipynb # # Notebook para preparar los datos de entrenamiento y testeo de los modelos a entrenar
│ ├── probar_modelo.ipynb # Notebook para probar el modelo entrenado (no obligatorio)
│ └── app_brain_tumor.py # Script o programa a ejecutar
│
├── .gitignore # Archivo para ignorar archivos/carpetas en Git
└── README.md # Archivo de documentación principal
```

## Orden de ejecución
Asegurese de ejecutar los notebooks en el siguiente orden
1. preparar_datos.ipynb
2. preparar_modelos.ipynb
3. entrenar_modelo.ipynb
4. En este punto ya puede ejecutar el programa app_brain_tumor.py


## Interfaz gráfica
El programa se ejecuta mediante una interfaz gráfica que consiste en una ventana con 2 botones:

- **Botón HOG**: Ejecuta un modelo entrenado con características HOG (Histogram of Oriented Gradients)
- **Botón LBP**: Ejecuta un modelo entrenado con características LBP (Local Binary Patterns)

Ambos botones permiten:
1. Seleccionar una imagen del conjunto de datos de prueba
2. Mostrar el diagnóstico en la parte superior de la ventana
