# Organización y conservación de archivos

## Proyecto

| Ruta | Función |
| --- | --- |
| `app.py` | Punto de entrada del demostrador; conservar en raíz. |
| `src/score_super_resolution/` | Implementación reutilizable. |
| `tests/` | Pruebas y fixtures pequeños. |
| `scripts/` | Generación, validación y reproducción de evidencias. |
| `notebooks/` | Interacción y ejecución local/Kaggle; el README distingue v1 de v2. |
| `configs/` | Protocolos y configuraciones congeladas, incluidos antecedentes v1. |
| `data/sources/` | Origen, revisión y licencia de datos. |
| `data/audits/` | Auditoría, exclusiones y revisiones. |
| `data/adaptation/` | Particiones por obra de la adaptación. |
| `data/manifests/` | Identidades y recuperación de manifiestos. |
| `data/schemas/` | Contratos de validación. |
| `data/raw/` | Datos locales no versionados; no publicar ni borrar como caché. |
| `artifacts/` | Evidencias generadas, métricas, revisiones y salidas locales. |
| `checkpoints/` | Pesos locales necesarios para inferencia; no versionados. |
| `assets/branding/` | Recursos del demostrador. |
| `docs/` | Protocolos, resultados, decisiones e índices. |

No se crea otra carpeta `datasets/`: `data/` es el único espacio de datos del proyecto.
La caché de Hugging Face y `.venv/` tienen funciones distintas y se conservan.

## Memoria (repositorio independiente)

`main.tex`, la clase, los estilos y los metadatos permanecen en raíz por compatibilidad
con Overleaf. `frontmatter/`, `chapters/`, `appendices/` y `ods/` contienen las fuentes;
`figures/` contiene recursos de publicación y sus previews. `docs/tfg-guidance/` conserva
la navegación académica; `scripts/` contiene la compilación y `build/` sus salidas locales.

Las figuras PDF son fuentes de publicación, no PDFs finales de compilación. Las previews PNG,
las capturas de revisión y todas las imágenes generadas se conservan como parte del trabajo,
aunque tengan otra representación en PDF o pertenezcan a una etapa anterior.

## Limpieza del 7 de septiembre de 2026

Ambos repositorios estaban sin cambios pendientes. Se revisaron los inventarios versionados,
archivos locales, referencias de figuras y convenciones de datos/notebooks. La separación de
repositorios y la exclusión de datos ya existían; esta limpieza no cambia las rutas experimentales.

Se apartaron cachés de Python, pytest y Ruff a una copia recuperable fuera del workspace,
con inventario de tamaños y SHA-256. Las previews y capturas inicialmente apartadas se restauraron
al aclarar el autor que todas las imágenes generadas forman parte del trabajo.
No se borraron resultados, pesos, paquetes Kaggle, datos, normativa, estados GSD ni verificaciones
históricas. Las cachés volverán a aparecer al ejecutar herramientas y seguirán ignoradas por Git.

No se consideran basura los experimentos fallidos o sustituidos: explican cambios del protocolo.
No se mueven configuraciones congeladas, notebooks ni artefactos sin actualizar y verificar todas
sus referencias. Los borradores administrativos se conservan hasta cerrar depósito y defensa.

La revisión de referencias encontró consumidores para los 26 módulos funcionales de `src/`.
Los scripts ejecutables sin referencias a su nombre incluyen generadores de figuras, revisiones
y congelación de muestras: su ausencia en imports no demuestra desuso. Incluso el borrador de
correo sustituido está referenciado por pruebas de decisiones. No se eliminó código por estos
indicios; esta revisión no equivale a demostrar que cada función se ejecutó en un experimento.

Validación de esta limpieza: las 51 imágenes apartadas se restauraron y sus SHA-256 coinciden;
no faltan referencias explícitas de imágenes en LaTeX y la compilación local permanece correcta.
Ruff y formato pasan para `src/`, `scripts/`, `tests/` y `app.py`. El lint global detecta 42
incidencias previas en notebooks, que no se modificaron. La batería completa se interrumpió
deliberadamente tras 12 minutos: 1057 pruebas pasaron, una fue omitida y no hubo fallos antes de
la interrupción; no se declara una ejecución completa satisfactoria.
