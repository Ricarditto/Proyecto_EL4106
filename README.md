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
3. Selección de variables: Cada pixel se tratará como una entrada independiente.

#### Algoritmo
 *Tipo de Algoritmo: Redes Neuronales Convolucionales (CNN) con diferentes arquitecturas (ResNet, DenseNet, EfficientNet y redes planas).
 *Justificación: Las conexiones residuales y densas permiten entrenar redes más profundas y eficientes al mitigar problemas como el desvanecimiento de gradientes y el uso excesivo de parámetros.

#### Definición de salidas
* Función objetivo: Maximizar la precisión de clasificación en el conjunto de validación.
* Principio de optimización: Utilización del optimizador Adam con una función de pérdida de entropía cruzada categórica.
* Criterio de detención: El entrenamiento se detendrá cuando la precisión en el conjunto de validación no mejore durante 10 épocas consecutivas (early stopping).

#### Parámetros por definir
* Hiperparámetros de las redes.
* Profundidad de las redes.
* Probabilidad de profundidad estocástica (en ResNet).
* Factores de escalamiento (en EfficientNet).

#### Software
- Frameworks: 
  * PyTorch para la implementación y entrenamiento de modelos.
  * TensorBoard para monitorear métricas durante el entrenamiento.
- Lenguaje de programación: Python.
- Aspectos programados:
  * Transformaciones y preprocesamiento de datos.
  * Entrenamiento de las arquitecturas seleccionadas.
  * Generación de gráficos y análisis comparativo.

#### Resultados esperados
1. Medidas de desempeño: Accuracy, loss, eficiencia computacional (memoria y tiempo).
2. Forma de presentación:
   * Tablas comparativas de precisión y pérdida en CIFAR-10 y MNIST.
   * Gráficos de curvas de entrenamiento y validación para cada modelo.
   * Análisis de eficiencia computacional para las arquitecturas evaluadas.

#### Estimación de recursos
1. La primera opción es utilizar Google Colab. Dada la limitación de recursos, se recomienda contar con los siguientes requerimientos en el entorno local:
   * GPU con al menos 8 GB de memoria para entrenar EfficientNet y DenseNet.
   * Almacenamiento: 10 GB para guardar modelos y resultados.
2. Número y tiempo de simulaciones:
   * Entre 5 y 10 simulaciones considerando diferentes combinaciones de hiperparámetros.
   * Tiempo estimado por simulación: 1-8 horas dependiendo del modelo y conjunto de datos.

#### Carta GANTT
| Semana | Tarea                                                                                     |
|--------|-------------------------------------------------------------------------------------------|
| 1      | Preprocesamiento y análisis exploratorio de datos.                                        |
| 2      | Implementación inicial de ResNet y redes planas.                                          |
| 4      | Entrenamiento y ajuste de hiperparámetros para ResNet.                                   |
| 8      | Implementación y entrenamiento de DenseNet.                                              |
| 10      | Implementación de EfficientNet y análisis de eficiencia computacional.                   |
| 13      | Comparación de resultados, análisis detallado y redacción de conclusiones.               |

#### Referencias
Ref: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
Ref: Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
Ref: He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
Ref: Huang, Gao, et al. "Densely Connected Convolutional Networks." CVPR. Vol. 1. No. 2. 2017.
Ref: Veit, Andreas, Michael J. Wilber, and Serge Belongie. "Residual networks behave like ensembles of relatively shallow networks." Advances in Neural Information Processing Systems. 2016.
Ref. Huang G., Sun Y., Liu Z., Sedra D., Weinberger K.Q. (2016) Deep Networks with Stochastic Depth. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9908. Springer, Cham. https://doi.org/10.1007/978-3-319-46493-0_39
Ref. M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” arXiv, 2019, doi: 10.48550/ARXIV.1905.11946.
Ref.  D. Balduzzi, M. Frean, L. Leary, J. Lewis, K. W.-D. Ma, and B. McWilliams, “The Shattered Gradients Problem: If resnets are the answer, then what is the question?,” arXiv, 2017, doi: 10.48550/ARXIV.1702.08591.
