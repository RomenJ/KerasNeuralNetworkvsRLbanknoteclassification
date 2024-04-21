El programa proporciona una implementación básica de un sistema de clasificación de billetes bancarios utilizando redes neuronales y regresión logística.

Comienza importando las bibliotecas necesarias, incluidas numpy, pandas, seaborn, matplotlib.pyplot, scikit-learn y TensorFlow. Estas bibliotecas son esenciales para cargar y procesar datos, construir modelos de aprendizaje automático y visualizar resultados.

El conjunto de datos se carga desde un archivo CSV llamado "banknotes.csv" utilizando pandas. Luego, se muestra información básica sobre el conjunto de datos, como su forma, las primeras 10 filas, la información del tipo de datos y estadísticas descriptivas.

Las características y etiquetas se definen separando las columnas de características del conjunto de datos. Luego, los datos se dividen en conjuntos de entrenamiento y prueba utilizando train_test_split de scikit-learn.

Se crea un modelo secuencial de red neuronal utilizando Keras y TensorFlow. El modelo consta de capas densas con funciones de activación ReLU y sigmoide. Se compila el modelo con una función de pérdida de entropía cruzada binaria y el optimizador Adam.

El modelo se entrena en el conjunto de entrenamiento durante 20 épocas. La precisión del modelo se grafica a lo largo de las iteraciones para visualizar el rendimiento durante el entrenamiento.

La precisión del modelo de red neuronal se evalúa en el conjunto de prueba utilizando métricas como la precisión, la matriz de confusión y el informe de clasificación.

Además, se entrena un modelo de regresión logística utilizando scikit-learn y se evalúa su precisión en el conjunto de prueba.

Se compara la precisión de ambos modelos mediante un gráfico de barras.

Finalmente, se calculan las probabilidades de predicción y se grafica la curva ROC para ambos modelos. La curva ROC y el área bajo la curva (AUC) proporcionan una medida de la capacidad de discriminación del modelo. Se muestra la curva ROC junto con la leyenda que indica el AUC para cada modelo.
