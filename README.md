# Superresolución aplicada a partituras digitalizadas

Este repositorio contiene el código y los experimentos asociados al Trabajo de Fin de Grado:

**“Estudio, adaptación y evaluación de técnicas de superresolución para la mejora de partituras digitalizadas”**

El objetivo principal del proyecto es estudiar hasta qué punto las técnicas actuales de superresolución permiten recuperar detalle y mejorar la calidad de partituras digitalizadas de baja resolución.

La memoria académica del TFG se mantiene en un repositorio independiente. Este repositorio está destinado al código, los experimentos, la preparación de datos, los modelos y las posibles herramientas desarrolladas durante el proyecto.

## Idea general

Las partituras presentan características visuales particulares: líneas finas de pentagrama, símbolos pequeños, texto, ligaduras, alteraciones y otros elementos cuya pérdida o modificación puede cambiar no solo la apariencia de la imagen, sino también su significado musical.

El proyecto parte de partituras disponibles en alta resolución que actuarán como referencia. A partir de ellas se generarán versiones degradadas de menor resolución mediante procesos controlados.

Sobre estas versiones se aplicarán distintas técnicas y modelos de superresolución. Los resultados obtenidos se compararán posteriormente con las imágenes originales para analizar hasta qué punto los modelos son capaces de reconstruir correctamente la información perdida.

El flujo experimental básico puede representarse de la siguiente manera:

`Partitura original → Degradación → Imagen de baja resolución → Superresolución → Comparación con el original`

## Alcance principal

El núcleo del TFG se centra en:

* Seleccionar conjuntos de datos adecuados de partituras digitalizadas.
* Diseñar uno o varios procesos controlados para generar imágenes de baja resolución.
* Estudiar y seleccionar técnicas y modelos actuales de superresolución potencialmente adecuados para este dominio.
* Aplicar modelos preentrenados como referencia inicial.
* Adaptar mediante *fine-tuning*, cuando resulte apropiado, los modelos seleccionados al dominio específico de las partituras.
* Comparar las reconstrucciones generadas con las imágenes originales de alta resolución.
* Evaluar los resultados mediante métricas de calidad de imagen y análisis visual.
* Analizar especialmente la conservación de elementos relevantes de la notación musical.

El objetivo no es necesariamente desarrollar una nueva arquitectura de superresolución, sino estudiar, adaptar y evaluar técnicas existentes en un dominio con características muy específicas.

## Metodología prevista

El planteamiento experimental parte de disponer de una imagen original de alta calidad que pueda utilizarse como referencia o *ground truth*.

A partir de dicha imagen se generará una versión degradada utilizando distintos procesos de reducción de resolución. Esta imagen será la entrada del modelo de superresolución.

La salida generada por el modelo podrá compararse directamente con el original, permitiendo estudiar cuantitativa y cualitativamente la capacidad del sistema para recuperar la información eliminada durante la degradación.

Además de métricas generales de reconstrucción de imagen, será especialmente importante observar posibles errores relevantes en una partitura, como:

* desaparición o interrupción de líneas;
* deformación de símbolos musicales;
* pérdida de elementos pequeños;
* unión de elementos originalmente separados;
* aparición de información inexistente en el original;
* alteraciones en texto o cifras;
* reconstrucciones visualmente plausibles pero musicalmente incorrectas.

## Posibles ampliaciones

El proyecto está diseñado de forma que el alcance pueda ampliarse si el desarrollo del núcleo principal avanza adecuadamente. Las siguientes líneas se consideran extensiones posibles y **no forman parte necesariamente del alcance mínimo del TFG**.

### Aplicación sobre imágenes y PDF

Una posible ampliación consiste en desarrollar una herramienta o prototipo que permita utilizar los modelos obtenidos de una forma más cercana a un caso de uso real.

Por ejemplo, podría permitir:

* cargar una imagen de una partitura;
* aplicar automáticamente el proceso de superresolución;
* visualizar o descargar el resultado;
* procesar páginas completas;
* aceptar documentos PDF;
* convertir las páginas del PDF en imágenes;
* aplicar superresolución a cada página;
* reconstruir posteriormente un PDF mejorado.

Esta extensión permitiría transformar los experimentos realizados durante el TFG en un prototipo funcional.

### Superresolución como preprocesamiento para OMR

Otra línea de ampliación consiste en estudiar si la mejora visual producida por la superresolución se traduce también en una mejora en sistemas de **Optical Music Recognition (OMR)**.

En este escenario se podrían comparar, para una misma partitura degradada:

`Imagen degradada → OMR`

frente a:

`Imagen degradada → Superresolución → OMR`

El objetivo sería comprobar si el uso de superresolución como etapa de preprocesamiento mejora el reconocimiento automático de los elementos musicales.

Este experimento permitiría complementar las métricas de calidad visual con una medida más relacionada con la utilidad real de la imagen reconstruida.

### Evaluación específica para notación musical

También podría estudiarse la incorporación de métricas o procedimientos de evaluación específicos para partituras.

Una imagen reconstruida puede presentar una buena similitud visual global respecto al original y, sin embargo, haber modificado un elemento musical pequeño pero semánticamente importante.

Por ello, una posible línea adicional consiste en estudiar métodos de evaluación que tengan en cuenta la estructura y el contenido musical, más allá de las métricas tradicionales de calidad de imagen.

### Otros experimentos

Dependiendo de los resultados obtenidos durante el desarrollo, también podrán explorarse aspectos como:

* diferentes tipos y niveles de degradación;
* diferentes factores de ampliación;
* comparación entre modelos generalistas y modelos adaptados a partituras;
* influencia del conjunto de entrenamiento utilizado;
* generalización entre diferentes tipos o estilos de partituras;
* comportamiento ante documentos escaneados reales;
* ruido, desenfoque, compresión u otras degradaciones habituales en archivos digitalizados.

## Estado del proyecto

Proyecto en desarrollo como parte de un Trabajo de Fin de Grado.

El alcance principal está definido alrededor de la **selección, adaptación y evaluación de técnicas de superresolución aplicadas a partituras digitalizadas**. Las funcionalidades adicionales descritas en este documento representan posibles líneas de ampliación y su implementación dependerá de la evolución y los resultados del proyecto.
