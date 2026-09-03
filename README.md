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

* Auditar SMB como benchmark principal y separar explícitamente el estudio principal de modelos
  preentrenados de cualquier adaptación secundaria del dominio.
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

## Organización del trabajo

El TFG se distribuye entre dos repositorios sincronizados con GitHub:

* `proyecto`: implementación, preparación de datos, configuraciones, experimentos, métricas y artefactos reproducibles;
* `memoria`: fuentes de la memoria académica, sincronizadas a su vez con el proyecto de Overleaf.

Las normas y recomendaciones académicas consolidadas se documentan en `memoria/docs/tfg-guidance/`. El contrato experimental y los criterios de reproducibilidad específicos del proyecto se mantienen en [`docs/research-protocol.md`](docs/research-protocol.md).

## Entorno de desarrollo

El entorno local utiliza **Python 3.12.12** y [`uv`](https://docs.astral.sh/uv/) como gestor de Python, dependencias y entorno virtual. Las versiones resueltas se conservan en `uv.lock`.

```bash
uv sync --extra cpu --group dev --group notebooks --group kaggle
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

El entorno local instala PyTorch para CPU y se utiliza para preparar datos, probar los pipelines, calcular métricas y ejecutar experimentos pequeños. Los entrenamientos o evaluaciones que necesiten aceleración se ejecutarán en Kaggle, registrando la revisión del *notebook*, el acelerador, el entorno, la configuración, las semillas y los artefactos de cada ejecución.

La guía completa está en [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Estructura del repositorio

* `src/score_super_resolution/`: código reutilizable del proyecto;
* `tests/`: pruebas deterministas sobre datos pequeños;
* `notebooks/`: exploración, análisis visual y comunicación de resultados;
* `configs/`: configuraciones versionadas de experimentos;
* `data/sources/`: descriptores versionados de las fuentes externas;
* `data/manifests/`: particiones y selecciones reproducibles; las imágenes no se almacenan en Git;
* `artifacts/`: convenciones para resultados y artefactos generados;
* `docs/research-protocol.md`: protocolo experimental y reglas de reproducibilidad.

## Reproducibilidad

Cada resultado que se utilice en la memoria deberá poder relacionarse con:

* una revisión concreta del código;
* el origen, versión, licencia y partición del conjunto de datos;
* la configuración de degradación y del modelo;
* las semillas aleatorias;
* una captura del entorno y del hardware;
* las métricas por elemento y agregadas;
* los logs y artefactos de salida.

Cuando haya que crear particiones, se definirán por partitura, obra o documento fuente antes de
generar páginas o parches, evitando que contenido relacionado aparezca simultáneamente en
entrenamiento, validación y prueba. Los *splits* oficiales de un benchmark no se redistribuirán
silenciosamente.

## Fuente principal de evaluación

La fuente principal será [Sheet Music Benchmark (SMB)](https://huggingface.co/datasets/PRAIG/SMB),
consumida directamente con Hugging Face Datasets y una revisión concreta, sin duplicarla dentro
del repositorio. El descriptor reproducible se encuentra en
[`data/sources/smb.yaml`](data/sources/smb.yaml).

El descriptor y la auditoría autenticada confirman que SMB requiere aprobación manual de acceso,
usa licencia CC BY-NC 4.0 y, en la revisión fijada, publica 685 páginas en un único *split* oficial
`test`. El estudio principal conserva ese rol: evalúa métodos preentrenados sobre 64 obras que no
se utilizan para ningún ajuste. Como estudio secundario y declarado separadamente, el proyecto
define particiones propias por obra dentro de las páginas restantes para adaptar EDSR al dominio:
45 obras/212 páginas de desarrollo previo para entrenamiento, 13 obras/35 páginas nuevas para
validación y 20 obras/55 páginas nuevas para test. Las 64 obras del estudio principal quedan
excluidas de las tres particiones. Este uso no se presenta como un *split* oficial de SMB ni como
evidencia de generalización fuera del corpus; el contrato completo está en
[`docs/smb-edsr-finetuning-v1.md`](docs/smb-edsr-finetuning-v1.md).

## Posibles ampliaciones

El proyecto está diseñado de forma que el alcance pueda ampliarse si el desarrollo del núcleo principal avanza adecuadamente. Las siguientes líneas se consideran extensiones posibles y **no forman parte necesariamente del alcance mínimo del TFG**.

### Demostrador profesional sobre imágenes

Se ha iniciado una ampliación acotada para utilizar el EDSR adaptado como generador de copias de
consulta ampliadas. El demostrador local permite cargar una imagen, escoger x2/x4, comparar el
original y el derivado, descargar la salida y conservar la identidad del modelo. Su contrato y sus
reglas de seguridad están en
[`docs/professional-demonstrator-v1.md`](docs/professional-demonstrator-v1.md).

El alcance actual permite:

* cargar una imagen de una partitura;
* aplicar el EDSR adaptado en x2 o x4;
* visualizar o descargar el resultado;
* procesar páginas completas;
* comparar el original y el derivado;
* mostrar tiempo, escala e identidad del checkpoint.

El tratamiento de PDF, el despliegue público y cualquier garantía de restauración permanecen fuera
de alcance. La ampliación solo se considerará resultado del TFG si completa la prueba externa y la
revisión antes del depósito.

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

La infraestructura inicial del repositorio ya está preparada: entorno reproducible con `uv`, Python 3.12.12, PyTorch local para CPU, soporte para Jupyter y Kaggle, pruebas, linting y captura del entorno de ejecución.

SMB está seleccionado y auditado en una revisión inmutable de Hugging Face. Una primera ejecución
piloto reveló que la degradación calibrada en píxeles absolutos no mantenía la severidad al cambiar
la escala de pentagrama. La evaluación principal final ya usa el
[protocolo v2 normalizado por pentagrama](docs/smb-protocol-v2.md) y una muestra nueva de 64 obras.
Tras completar ese núcleo, se ha promovido un estudio secundario acotado de
[*fine-tuning* de EDSR](docs/smb-edsr-finetuning-v1.md), con particiones propias por obra y test
nuevo. La ejecución acelerada ya está reconciliada: los resultados y sus límites se documentan en
[`docs/smb-edsr-finetuning-v1-results.md`](docs/smb-edsr-finetuning-v1-results.md) y se incorporan
a la memoria sin publicar imágenes ni pesos derivados de SMB. Las
funcionalidades adicionales continúan siendo opcionales salvo el demostrador de imágenes y su
prueba externa acotada, promovidos bajo un protocolo removible que no puede retrasar el depósito.
