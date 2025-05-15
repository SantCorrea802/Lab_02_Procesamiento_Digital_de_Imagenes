# Laboratorio 2 - Procesamiento Digital de Imágenes 2025-1 UdeA

## Set de datos utilizado
[Brain Tumor Dataset en Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

## Descripción
Este proyecto utiliza el set de datos *Brain Tumor Dataset* para detectar cuando una radiografía de cerebro presenta un tumor (clasificado como `1`) o cuando no presenta tumor (clasificado como `0`).

## Interfaz gráfica
El programa se ejecuta mediante una interfaz gráfica que consiste en una ventana con 2 botones:

- **Botón HOG**: Ejecuta un modelo entrenado con características HOG (Histogram of Oriented Gradients)
- **Botón LBP**: Ejecuta un modelo entrenado con características LBP (Local Binary Patterns)

Ambos botones permiten:
1. Seleccionar una imagen del conjunto de datos de prueba
2. Mostrar el diagnóstico en la parte superior de la ventana
