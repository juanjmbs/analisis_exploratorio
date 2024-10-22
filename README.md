# Análisis de la Base de Datos y Evaluación de Modelos Predictivos

## Introducción

Este proyecto se centra en el análisis de una base de datos de departamentos en venta, con el objetivo de identificar la composición de los datos, detectar datos nulos y outliers, y evaluar modelos predictivos tanto con datos procesados como sin procesar.

## Contenido

1. Análisis de la Base de Datos
2. Evaluación de Modelos Predictivos
3. Procesamiento de Datos con y sin Outliers
4. Test de Normalidad
5. Minería de texto (Spacy-Lematizador-Stopwords)
6. N-gramas

## Análisis de la Base de Datos

Se visualizan los datos para entender su conformación y encontrar parámetros que permitan identificar su composición. Se verifica la existencia de datos nulos y outliers.

## Evaluación de Modelos Predictivos

Se utilizan diversas técnicas de machine learning para evaluar modelos predictivos. A continuación, se muestra un ejemplo de cómo se implementa un modelo de regresión de árbol de decisión:

```python
import sklearn as sk
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import load_iris

plt.rcParams['figure.figsize'] = [30, 10]

# Preparación de los datos
X = Dptos_en_venta2[Dptos_en_venta2.columns.drop("price")]
y = Dptos_en_venta2["price"]

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
reg = sk.tree.DecisionTreeRegressor(max_depth=3, criterion="friedman_mse", random_state=42)
reg.fit(X_train, y_train)

# Predicción de los valores de prueba
y_pred = reg.predict(X_test)

# Evaluación del modelo
mse = sk.metrics.mean_squared_error(y_test, y_pred, squared=False)
print(f"Error cuadrático medio: {mse}")
