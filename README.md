# portafolio-implementacion
En este repositorio, se implementa un modelo para la predicción de la masa de pingüinos de acuerdo a sus características fisiológicas, así como sus características categóricas. 

## Estructura del proyecto

Este diagrama muestra la organización de carpetas y archivos del proyecto, junto con su descripción de contenido.
```text
portafolio-implementacion/
│
├── main.py                                         # Archivo principal (ejecución de todo el código)
├── README.md                                       # Descripción de la estructura del proyecto
├── transformation.py                               # Archivo de transformación y limpieza de datos     
├── Reporte_PreedicciónPingüinos_A01705541.pdf
│
├── data/
│   └── penguins_size.csv                           # Archivo .csv del conjunto de datos completo
│
├── algorithms/
│   ├── cross_validation.py                         # Algoritmo de cross-validation
│   └── regression_gd.py                            # Algoritmo de gradient descent para regresión lineal                  
│
├── visualization/
│   ├── statistic.py                                # Archivo con las funciones para gráficos utilizadas en el análisis
│   └── visualization.py                            # Archivo que llama todas las funciones necesarias de statistic.py
│
└── .gitignore
```