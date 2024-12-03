# Proyecto Inteligencia Computacional - EL4106
## Estudio del efecto de la profundidad y conexiones residuales en redes neuronales
### Autores: Benito Fuentes, Ricardo Salas.
### Tutor: Andrés González

#### Descripción del problema y motivación
El proyecto busca analizar cómo las arquitecturas profundas con conexiones residuales (ResNet) y densas (DenseNet) impactan el desempeño de redes neuronales convolucionales. Estas arquitecturas han demostrado superar limitaciones tradicionales como el desvanecimiento de gradientes y el uso ineficiente de parámetros. La motivación principal es optimizar el diseño de modelos para aplicaciones que requieren precisión y eficiencia computacional, abordando desafíos en tareas complejas como el reconocimiento de imágenes.

#### Objetivos del proyecto
* Comparar redes planas y ResNet en términos de precisión y eficiencia.
* Evaluar el impacto de la profundidad estocástica en ResNet.
* Estudiar las arquitecturas DenseNet y EfficientNet, destacando su eficiencia en el uso de recursos.
* Proponer recomendaciones sobre la arquitectura más adecuada para diferentes escenarios prácticos.

#### Base de Datos
Se utilizarán dos conjuntos de datos:
* CIFAR-10: Imágenes a color de 10 clases distintas, adecuado para medir el desempeño en problemas complejos de clasificación.
* MNIST: Dígitos escritos a mano, utilizado para validar la generalización de los modelos en un entorno más sencillo.

#### Preprocesamiento de Datos
1. Transformaciones:
  * Escalado de imágenes a un rango [0, 1].
  * Normalización según la media y desviación estándar del conjunto de entrenamiento.
2. Augmentación: Rotaciones, recortes y volteos para mejorar la capacidad de generalización.
3. Selección de Variables: Cada pixel se tratará como una entrada independiente.
