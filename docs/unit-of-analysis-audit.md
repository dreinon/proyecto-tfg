# SMB: auditoría de la unidad de análisis

Fecha: 7 de septiembre de 2026. Nota de interpretación posterior a la ejecución; no cambia
experimentos, particiones, configuraciones, semillas, pesos ni resultados congelados.

## Unidad implementada

`source_group_id` se obtiene de la identidad normalizada `original_score` eliminando únicamente
el sufijo terminal `_pN`, donde N es un número de página. La implementación está en
[`_canonical_source_group_id`](../src/score_super_resolution/smb_audit.py), con el patrón
`_p[0-9]+\Z`. No elimina sufijos de movimiento ni reconcilia alias, composiciones o ediciones.
Por ello, «grupo documental» o «ID de fuente» describe lo comprobado; «obra completa independiente»
es una garantía más amplia que estos identificadores no acreditan.

| Ámbito | Grupos documentales | Páginas |
| --- | ---: | ---: |
| Auditoría SMB | 260 | 685 |
| Evaluación principal v2 | 64 | 64 seleccionadas |
| Adaptación: entrenamiento | 45 | 212 |
| Adaptación: validación | 13 | 35; 13 representantes |
| Adaptación: prueba | 20 | 55; 20 representantes |

Las doce obras del piloto externo pertenecen a otra colección y conservan su denominador propio;
esta corrección de la agrupación SMB no las convierte en grupos del benchmark ni modifica sus
216 resultados.

## Relaciones observadas

El [manifiesto adaptativo congelado](../data/adaptation/smb-edsr-finetuning-v1-split.csv) incluye
estos ejemplos verificables:

| Composición identificable | Entrenamiento | Otro rol |
| --- | --- | --- |
| Beethoven, sonata 01 | `beethoven_sonata01_2` | prueba: `beethoven_sonata01_4` |
| Mozart, sonata 14 | `mozart_sonata14_2` | validación: `mozart_sonata14_1` |
| Mozart, sonata 15 | `mozart_sonata15_1` | prueba: `mozart_sonata15_3` |

Son IDs distintos de movimientos de una misma composición. La muestra principal contiene también
varios movimientos de algunas sonatas. Compartir compositor, por sí solo, no se considera fuga.
No se ha completado un mapa bibliográfico de alias, ediciones y composiciones que permita dar un
número definitivo de obras completas independientes.

## Qué se verificó y qué no

La verificación automatizada separada del pipeline principal confirma ausencia de páginas e IDs
idénticos compartidos entre los tres roles adaptativos y ausencia de los 64 IDs principales en
ellos. Los 45 IDs de entrenamiento proceden de los 53 del piloto de desarrollo. Esto no prueba
reutilización de los mismos píxeles entre entrenamiento y prueba, pero tampoco descarta dependencia
entre movimientos relacionados ni acredita generalización a composiciones completas nunca vistas.

Los intervalos bootstrap publicados remuestrean 64 grupos principales o 20 grupos de prueba
adaptativa según el estudio. Son condicionales a tratar esos grupos como independientes. Su
recomputación correcta verifica el cálculo, no esa independencia estadística; la correlación
residual entre movimientos o ediciones puede afectar a la precisión de los intervalos. Las medias
y mejoras describen los documentos efectivamente evaluados bajo las degradaciones congeladas.

## Conservación y uso futuro

Los [protocolos v2](smb-protocol-v2.md) y [adaptativo](smb-edsr-finetuning-v1.md) conservan su
redacción histórica con un aviso fechado. Esta nota y el [protocolo activo](research-protocol.md)
delimitan su interpretación actual sin atribuir controles que no se implementaron.

Para sostener futuras conclusiones por obra completa, se debe validar primero el mapa de parentesco
musical y separar los clústeres antes de extraer páginas o parches. Un análisis de sensibilidad
sobre resultados existentes debe identificarse como posterior a la ejecución, conservar el cálculo
original y declarar sus reglas de agrupación. Esta nota no afirma que dicho análisis se haya hecho.
