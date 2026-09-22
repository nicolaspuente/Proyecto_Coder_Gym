# GymIA — datos que impulsan decisiones

Proyecto final de **Nicolás Puente** · Inteligencia Artificial: Generación de Prompts · Comisión #96165 · Coderhouse (2026).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolaspuente/Proyecto_Coder_Gym/blob/main/GymIA_Proyecto_Final_Colab.ipynb)

> **Estado:** POC incorporada a `main`. El enlace de Colab abre el cuaderno con los resultados y la comparación de prompts. Falta registrar el modelo exacto usado en las pruebas manuales si se desea una comparación controlada.

## Resumen

GymIA transforma los registros de un gimnasio de Mendoza en indicadores calculados, una propuesta de acción supervisada y una pieza visual conceptual. La prueba combina Python/Pandas con técnicas de prompting **zero-shot**, **one-shot**, prompting dirigido y revisión iterativa. Evita atribuir causas o cobrar importes que no constan en los datos.

## Problema, objetivo y alcance

Un archivo contiene una fila por actividad del socio y por mes: contar filas como personas distorsiona los indicadores. El objetivo de la POC es validar la unidad de análisis, detectar un cambio mensual significativo y comunicar una recomendación con límites explícitos. No sustituye una decisión comercial ni predice bajas; no contiene asistencia, motivos de inactividad o comprobantes de cobro.

## Fuentes y metodología

- `DATASETGYM.csv`: 4.374 registros de socio, actividad y mes.
- `Socios_Mes.csv`: 4.174 registros únicos de socio y mes, para 650 ID únicos en 2024.
- El cuaderno compara ambos archivos en una revisión fija y verifica cantidades de actividades, estados, suma de cuotas/inscripciones y totales.
- Hay 63 grupos socio-mes con actividades en estados diferentes. El consolidado aplica «Activo si al menos una actividad está activa»; **regla de negocio confirmada por el autor el 22/09/2026**.
- Se calculan indicadores en Pandas antes de pasarlos a los prompts. No se envían nombres, edades, localidades ni identificadores personales al modelo.
- Se comparan un prompt directo sin ejemplos y otro con estructura, restricciones, un ejemplo de salida y manejo de ambigüedades. El autor aportó las dos respuestas observadas y se evaluaron con una rúbrica de seis criterios. El modelo exacto no quedó registrado.
- Un prompt de texto a imagen creó la [campaña conceptual](assets/campana_reactivacion_gymia.jpg). Se conserva el prompt exacto en el cuaderno.

## Resultados calculados

| Indicador | Mayo 2024 | Junio 2024 |
| --- | ---: | ---: |
| Socios registrados | 257 | 311 |
| Socios activos | 219 | 187 |
| Socios inactivos | 38 | 124 |
| Importes registrados (ARS) | 7.735.000 | 6.855.000 |

Entre socios presentes en ambos meses, **105 pasaron de activo a inactivo** y **19 de inactivo a activo**. La cantidad total de socios creció mientras bajaron los activos. Esto justifica investigar el cambio y ensayar una acción de reactivación. Los datos no permiten afirmar su causa ni medir el impacto de una campaña aún no ejecutada.

## Cómo ejecutar

1. Abrir `GymIA_Proyecto_Final_Colab.ipynb` en Google Colab y ejecutar las celdas en orden. El cuaderno descarga los CSV de una revisión fija de este repositorio.
2. Revisar validaciones e indicadores.
3. Revisar las dos respuestas manuales aportadas por el autor y la rúbrica ya completada. Para repetir la comparación con un modelo controlado, cambiar `RUN_API=True` en la celda opcional; se solicita la clave de forma interactiva y se hacen **dos llamadas** que pueden generar costo. No publicar claves.
4. Si se repite la prueba, registrar modelo, fecha, respuestas y costo; evaluar las nuevas respuestas por separado.
5. Revisar la imagen conceptual, el prompt que la generó y sus limitaciones.

## Resultados y conclusiones

Se validó la consistencia aritmética entre ambas tablas y se evitó confundir filas de actividad con socios. La prueba encontró un cambio concreto entre mayo y junio. La pieza visual generada es coherente con una invitación a retomar actividad, pero no demuestra eficacia comercial.

Las dos respuestas observadas respetan las cifras principales, la regla de estado y los límites del dataset. La rúbrica da **5/6 al prompt base** y **6/6 al dirigido con ejemplo**: la respuesta base propone contactar a los socios, pero no define una métrica; la mejorada propone auditar los 105 casos y medir cuántos quedan con motivo documentado. Es una comparación de dos respuestas, sin identificación del modelo exacto ni réplicas; no permite afirmar superioridad general de la técnica. Los 63 casos de estados mixtos abarcan todo 2024 y no se deben atribuir automáticamente al grupo de 105 transiciones de mayo a junio.

## Referencias

- Material de clase y consignas de «Inteligencia Artificial: Generación de Prompts», Coderhouse, comisión #96165.
- [Documentación oficial de OpenAI: primera llamada a API](https://platform.openai.com/docs/quickstart/make-your-first-api-request).
- Fuente de datos: los dos CSV de este repositorio.

**Nota sobre datos:** el autor confirmó el 22/09/2026 que los CSV son sintéticos y autorizó su publicación. La POC no necesita nombres ni edades y descarta esas columnas al leerlas.
