# ENTEL-RETO-2

## Integrantes

* Jose Chipana Tica

## Carpertas y archivos

#### EDA

* La carpeta EDA contiene tres notebooks que representan la mirada que he dado a los datos antes de crear los modelos,
  así como encontrar patrones que puedan ayudar.

#### Notebooks

* En esta carpeta se encuentran los diferentes notebooks en los que he probado diferentes modelos e ingeniería de datos
  para lograr los mejores resultados.

#### utils

* Contiene los scripts para la ejecución en un enterno de pruebas para producción
* Desde la extracción de datos hasta la creación de modelos

### Ejecución

* El notebook "pipeline" contiene todos los scripts para ejecutar de una manera rápida, mientras que el script "
  pipeline" ejecuta el mismo proceso pero ya para un entorno de producción

* Para ejecutar el script o notebook es necesario el uso de GPU debido a la alta demanda de recurso que utilizan las
  técnicas de LSTM

## Técnicas

### Featuring

* Para generar más variables que puedan influenciar en los modelos me basé en un notebook de una competición de
  forecasting: https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution de dónde entendí que los datos en
  función en diferentes tiempo puede captar mayor información

### LSTM

* La primera técnica que se me vino a la mente fue LSTM debido a que esta técnica tiene la capacidad de retener datos
  con el paso de tiempo y que estos influyan en el futuro.
* Para utilizar esta técnica me base en un notebook de un Kaggle
  Granmaster como es de Chris Deotte:  "https://www.kaggle.com/code/cdeotte/lstm-feature-importance"

### MLP

* Multilayer perceptron es una técnica que no capta la variable del tiempo, es decir, que trata a todas las variables
  como iguales. La variable Semana0 y la variable Semana50 tiene la misma influencia sin importar si una es la primera y
  la otra la última.

* Utilicé este tipo de técnica para captar los patrones sin que la linealidad del tiempo influya, así evitar crear
  modelos similiares para luego unificarlos.