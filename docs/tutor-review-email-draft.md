# Near-final joint review email draft

Status: Draft — not sent

This repository file is preparation only. Sending remains a student action and no response is
inferred here.

## Elena Vázquez Barrachina y Jorge Calvo Zaragoza — revisión conjunta

**Asunto:** Borrador completo del TFG para revisión (entrega 7 de septiembre)

Hola Elena y Jorge:

Os adjunto un borrador completo y estable de la memoria y de la evaluación final sobre SMB. El
trabajo compara interpolación bicúbica, EDSR y SwinIR en x2/x4 con tres niveles de degradación
normalizados por la escala del pentagrama. Además, incorpora una adaptación acotada de EDSR con
particiones independientes por obra y conserva un test intacto. La adaptación mejora todas las
condiciones evaluadas frente al EDSR preentrenado, aunque la revisión visual confirma que la
información pequeña perdida bajo degradación fuerte no se recupera de forma fiable y no debe
presentarse como restauración musical automática.

La versión que os comparto ya está completa y compilada; corresponde a la revisión
`780cdea`. También incluye una ampliación profesional acotada ya cerrada: un demostrador local del
modelo y una prueba sin reajuste sobre doce obras externas. En las 216 salidas, el modelo adaptado
mejoró en promedio al EDSR oficial bajo las seis condiciones; la revisión fija consideró ocho
derivados aceptables para consulta, tres aceptables con reservas y uno no consultable. La memoria
explicita que las LR se generaron sintéticamente, que revisó los casos un único evaluador y que esto
no demuestra restauración de LR naturales ni corrección musical universal.

Cuando podáis, ¿podríais revisar el trabajo completo, incluida la estructura, el diseño
experimental, la interpretación de los resultados, la adecuación del alcance, los límites de las
conclusiones y su conexión con usos profesionales? Podéis empezar sobre esta versión desde ahora;
cualquier cambio posterior quedará limitado a la ampliación indicada y a las correcciones que me
trasladéis.

He reservado los días siguientes para incorporar vuestras indicaciones y realizar la comprobación
final en Overleaf y Ebrón. No está previsto añadir más alcance antes del depósito.

Gracias.
