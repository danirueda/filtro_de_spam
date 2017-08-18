# Filtro de spam
### Resumen
Trabajo en el que se evalúan distintas métricas de un clasificador de correo
electrónico construyendo este con distintas distribuciones de probabilidad.

### Datos técnicos
Para entrenar y probar el clasificador, se ha usado la base de datos pública de
correos electrónicos [Enron-Spam](http://www2.aueb.gr/users/ion/data/enron-spam/),
diseñada para el entrenamiento y evaluación de sistemas de filtrado de spam.

Se ha usado la distribución de Python [Anaconda](https://www.continuum.io/downloads)
en su versión 3.6.1 junto con el paquete de aprendizaje automático
[Scikit-Learn](http://scikit-learn.org/stable/).

### Breve descripción del trabajo
Para entrenar el clasificador, se ha usado el algoritmo de validación cruzada
K-fold cross-validation, probando con distintos folds.

A la hora de evaluar el clasificador se han utilizado 3 métricas:
1. La curva de precisión-recall.
2. La matriz de confusión.
3. F1-score.

El entrenamiento y la clasificación se han realizado con todas las combinaciones
de distribuciones de probabilidad (Multinomial o Bernoulli) normaliza o sin normalizar y
modelo de bolsa de palabras (unigramas o bigramas). De una forma mas detallada:

* Multinomial con unigramas.
* Multinomial con bigramas.
* Multinomial normalizada con unigramas.
* Multinomial normalizada con bigramas.
* Bernoulli con unigramas.
* Bernoulli con bigramas.
* Bernoulli normalizada con unigramas.
* Bernoulli normalizada con bigramas.

## Información del repositorio
El fichero `filtro_spam.py` contiene el código Pyhton para hacer el entrenamiento
y validación de los clasificadores.

El fichero `informe.pdf` es un informe escrito que contiene aspectos teóricos del
trabajo y el análisis de los resultados obtenidos.

## Puesta en marcha
Es necesario tener instalada la distribución de Python Anaconda y el paquete de
aprendizaje automático Scikit-Learn mencionado anteriormente.

Para lanzar el programa, simplemente hay que ubicarse en el directorio donde se
encuentre el fichero `filtro_spam.py` e introucir como parámetros la ruta
donde se encuentran las carpetas de los mails Enron (con '/' al final) y el número de
folds con el que se quiere ejecutar el algortimo de entrenamiento K-fold cross-validation.

```
$ python filtro_spam.py /ruta/hasta/mails/enron/ numero_de_folds
```

**Nota:** Dependiendo del sistema operativo, los separadores de directorio cambian
de /  a \ y viceversa. Por ello igual es necesario cambiar al separador correspondiente
en las líneas 42 y 77 de la función `load_enron_folder(path)` del fichero `filtro_spam.py`.