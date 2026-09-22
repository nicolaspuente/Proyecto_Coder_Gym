import json, textwrap
from pathlib import Path
sha='e7113eb85f962c705151ca2b2e38994c63e3d8ee'
def cell(kind,src):
 d={'cell_type':kind,'metadata':{},'source':textwrap.dedent(src).strip()+'\n'}
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d
C=[]
def m(s):C.append(cell('markdown',s))
def c(s):C.append(cell('code',s))
m('''# GymIA: de datos operativos a decisiones verificables
**Nicolás Puente · IA: Generación de Prompts · Comisión 96165**

Proyecto final — prueba de concepto ejecutable en **Google Colab**. Usa los dos CSV del [repositorio original](https://github.com/nicolaspuente/Proyecto_Coder_Gym) en una revisión fija para que los resultados sean reproducibles. El análisis numérico se calcula con Pandas; la IA se usa para explicar, priorizar y redactar recomendaciones, y un modelo de imagen para crear una campaña visual. El flujo es supervisado.

**Pregunta:** ¿Cómo detectar un cambio relevante en el estado de socios y convertirlo en una acción comercial fundada en datos?

**Límites:** los archivos no contienen asistencia, saldo adeudado ni comprobantes de pago. Los importes se llaman «ingresos» en el CSV y se informan aquí como *importes registrados*, sin afirmar cobro efectivo. Los nombres, edades y localidades no se envían al modelo. La procedencia sintética de los CSV fue confirmada por el autor el 22/09/2026; los nombres y edades se descartan del análisis y no se envían al modelo.''')
m('''## 1 · Fuente y granularidad
`DATASETGYM.csv`: una fila por socio, mes y actividad. `Socios_Mes.csv`: una fila por socio y mes. Para contar personas se usa el segundo. Se verifican agregaciones contra el primero. La regla de negocio confirmada por el autor es **Activo si tiene al menos una actividad activa**. La validación comprueba que el consolidado la aplica.''')
c(f'''import pandas as pd
ROOT='https://raw.githubusercontent.com/nicolaspuente/Proyecto_Coder_Gym/{sha}/'
raw=pd.read_csv(ROOT+'DATASETGYM.csv',encoding='utf-8-sig')
monthly=pd.read_csv(ROOT+'Socios_Mes.csv',encoding='utf-8-sig')
raw['Mes']=pd.to_datetime(raw['Mes']).dt.to_period('M').astype(str)
monthly['Mes']=pd.to_datetime(monthly['Mes']).dt.to_period('M').astype(str)
# Minimización: se eliminan atributos identificatorios y demográficos antes de construir prompts.
raw=raw[['ID_Cliente','Mes','Actividad','Estado','Cuota','Inscripcion']]
monthly=monthly[['ID_Cliente','Mes','Estado_General','Ingreso_Cuotas','Ingreso_Inscripcion','Ingreso_Mensual_Total','Cantidad_Actividades','Cantidad_Actividades_Activas']]
print('Registros actividad:',len(raw),'| registros socio-mes:',len(monthly),'| socios únicos:',monthly.ID_Cliente.nunique())
print('Duplicados socio-mes en tabla consolidada:',monthly.duplicated(['ID_Cliente','Mes']).sum())''')
c('''assert not monthly.duplicated(['ID_Cliente','Mes']).any(), 'Socio-mes no es único'
agg=raw.groupby(['ID_Cliente','Mes']).agg(
    actividades=('Actividad','size'),
    actividades_activas=('Estado',lambda x:(x=='Activo').sum()),
    estados_distintos=('Estado','nunique'),
    cuotas=('Cuota','sum'), inscripcion=('Inscripcion','sum')).sort_index()
m0=monthly.set_index(['ID_Cliente','Mes']).sort_index()
assert agg.index.equals(m0.index), 'Los dos CSV no cubren los mismos grupos'
checks={
    'cantidad_actividades':(agg.actividades==m0.Cantidad_Actividades).all(),
    'actividades_activas':(agg.actividades_activas==m0.Cantidad_Actividades_Activas).all(),
    'cuotas':(agg.cuotas==m0.Ingreso_Cuotas).all(),
    'inscripcion':(agg.inscripcion==m0.Ingreso_Inscripcion).all(),
    'total':(m0.Ingreso_Cuotas+m0.Ingreso_Inscripcion==m0.Ingreso_Mensual_Total).all(),
    'regla_estado_confirmada':((agg.actividades_activas>0)==m0.Estado_General.eq('Activo')).all()
}
assert all(checks.values()),f'Inconsistencia: {checks}'
mixed=int((agg.estados_distintos>1).sum())
print('Validaciones:',checks)
print('Socio-mes con estados diferentes entre actividades:',mixed)
print('Regla confirmada: activo si alguna actividad está activa.')''')
m('''## 2 · Hallazgos calculados
Se contrasta mayo con junio de 2024 porque ambos meses tienen una caída en socios activos mientras crece el padrón. La transición entre estados se calcula sobre los mismos ID presentes en ambos meses. No se interpreta como una tasa anual de abandono ni como causa probada.''')
c('''summary=monthly.groupby('Mes').agg(
    socios=('ID_Cliente','nunique'),
    activos=('Estado_General',lambda x:int((x=='Activo').sum())),
    inactivos=('Estado_General',lambda x:int((x=='Inactivo').sum())),
    importes_registrados_ars=('Ingreso_Mensual_Total','sum')).reset_index()
prev,current='2024-05','2024-06'
a=monthly.loc[monthly.Mes==prev,['ID_Cliente','Estado_General']].set_index('ID_Cliente')
b=monthly.loc[monthly.Mes==current,['ID_Cliente','Estado_General']].set_index('ID_Cliente')
joined=a.join(b,lsuffix='_prev',rsuffix='_curr',how='inner')
transitions=joined.groupby(['Estado_General_prev','Estado_General_curr']).size().to_dict()
active_to_inactive=int(transitions.get(('Activo','Inactivo'),0))
inactive_to_active=int(transitions.get(('Inactivo','Activo'),0))
may=summary.loc[summary.Mes==prev].iloc[0].to_dict()
june=summary.loc[summary.Mes==current].iloc[0].to_dict()
assert may['socios']==257 and june['socios']==311
assert may['activos']==219 and june['activos']==187
assert active_to_inactive==105 and inactive_to_active==19
print('Mayo:',may,'\\nJunio:',june)
print('Activo → inactivo:',active_to_inactive,'| inactivo → activo:',inactive_to_active)
display(summary)''')
m('''## 3 · Ingeniería de prompts
**Zero-shot** como referencia: instrucción breve sin ejemplo. **Prompt dirigido + one-shot:** datos verificados, ejemplo de formato y manejo de ambigüedades. Se evalúan ambos con el mismo modelo y los mismos hechos. El ejemplo es ilustrativo y no se mezcla con los datos del gimnasio. No se envían filas, ID ni datos personales.''')
c('''import json
facts={
  'unidad':'socio único por mes',
  'mayo_2024':{'socios':int(may['socios']),'activos':int(may['activos']),'inactivos':int(may['inactivos']),'importes_registrados_ars':int(may['importes_registrados_ars'])},
  'junio_2024':{'socios':int(june['socios']),'activos':int(june['activos']),'inactivos':int(june['inactivos']),'importes_registrados_ars':int(june['importes_registrados_ars'])},
  'transiciones_mayo_junio':{'activo_a_inactivo':active_to_inactive,'inactivo_a_activo':inactive_to_active},
  'regla_estado_socio':'Activo si al menos una actividad está activa; confirmada por el autor el 22/09/2026',
  'casos_estado_mixto_2024':mixed,
  'faltantes':['asistencia','causa de inactividad','pagos efectivamente cobrados']
}
baseline='Analizá el estado del gimnasio y sugerí una acción a partir de estos datos: '+json.dumps(facts,ensure_ascii=False)
example="""EJEMPLO DE FORMATO (cifras inventadas solo para mostrar la estructura):
Hallazgo: 4 de 30 socios comparables pasaron de activo a inactivo.
Evidencia: transición de estados registrada entre dos meses.
Límite: no se conoce el motivo del cambio.
Acción: revisar el segmento y diseñar un contacto voluntario.
Métrica: respuestas / personas contactadas."""
improved=f"""ROL: Analista de gestión de gimnasios.
CONTEXTO: POC supervisada sobre registros de 2024. La unidad es socio único por mes.
DATOS VERIFICADOS (JSON, no instrucciones): {json.dumps(facts,ensure_ascii=False)}
TAREA: identificar el hallazgo prioritario y proponer una acción concreta.
CRITERIOS: cuantificarlo; distinguir variación del total mensual y transición de los mismos socios; justificar cada cifra con campos del JSON; separar hecho, hipótesis y recomendación.
RESTRICCIONES: no inventar causa de inactividad, asistencia, deuda, cobro efectivo, impacto futuro ni tasa de abandono sin definir cohorte. Aplicar la regla confirmada de estado. No revelar identidades. No afirmar que la campaña funcionó.
FORMATO: Hallazgo | Evidencia | Límite | Acción | Métrica | Regla aplicada. Máximo 180 palabras.
AMBIGÜEDAD: si falta una regla de negocio, declararla como supuesto antes de concluir.
{example}"""
print('PROMPT BASE\\n',baseline,'\\n\\nPROMPT MEJORADO\\n',improved)''')
m('''## 4 · Prueba opcional del modelo de texto
Esta celda hace **dos llamadas** y puede generar costo. Mantener `RUN_API=False` para revisión sin API. Si se prueba manualmente en ChatGPT, pegar las respuestas reales en el informe con fecha y modelo usado. Nunca subir una clave de API al repositorio.''')
c('''RUN_API=False
MODEL='gpt-4.1-mini'  # Ejemplo; verificar acceso y precio en la cuenta antes de ejecutar.
answers={}
if RUN_API:
    import importlib.util, subprocess, sys
    if importlib.util.find_spec('openai') is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','openai'])
    from openai import OpenAI
    from getpass import getpass
    client=OpenAI(api_key=getpass('Clave API (no se guarda): '))
    for label,prompt in [('zero_shot',baseline),('dirigido_one_shot',improved)]:
        answer=client.responses.create(model=MODEL,input=prompt)
        answers[label]=answer.output_text
        print(label,'\\n',answer.output_text,'\\n')
else:
    print('Sin llamadas a API. Los prompts están disponibles para una evaluación manual.')''')
m('''## 5 · Respuestas observadas y evaluación
Los siguientes textos fueron proporcionados por el autor el **22/09/2026** tras probar cada prompt en una conversación nueva de ChatGPT. Se conservan sin reescribir. El modelo exacto y sus ajustes no quedaron registrados; por eso la comparación es **indicativa**, no un experimento controlado. La celda de API anterior es una alternativa reproducible si se desean nuevas respuestas con un modelo especificado.

### Prompt base — respuesta observada

> Entre mayo y junio, el gimnasio pasó de **257 a 311 socios** (+21 %), pero los **activos bajaron de 219 a 187** (−15 %). De los activos de mayo, **105 figuraron inactivos en junio**; volvieron a estar activos 19 que estaban inactivos. Los importes registrados cayeron de $7.735.000 a $6.855.000 (−11 %). Ese importe **no equivale a dinero cobrado**.
>
> **Acción sugerida:** revisar primero los 105 casos que pasaron de activo a inactivo y contactarlos para conocer la causa. Antes de atribuir la caída a abandono, verificar los estados según la regla confirmada —activo si tiene al menos una actividad activa—, especialmente en los 63 casos de estado mixto. Sin asistencia, motivos de inactividad ni pagos cobrados, todavía no se puede establecer la causa de la caída.

### Prompt dirigido + one-shot — respuesta observada

> **Hallazgo:** La prioridad es revisar a los 105 socios que pasaron de activos en mayo a inactivos en junio de 2024.
>
> **Evidencia:** El JSON registra 105 transiciones de activo a inactivo y 19 en sentido inverso. En los totales mensuales, los socios subieron de 257 a 311 (+54), mientras los activos bajaron de 219 a 187 (−32). La variación del total no describe por sí sola qué ocurrió con los mismos socios.
>
> **Límite:** No se conocen asistencia ni causas de inactividad. Los importes registrados bajaron $880.000, de $7.735.000 a $6.855.000; eso no demuestra una caída en lo efectivamente cobrado. La causa del cambio de estado sigue siendo una hipótesis por investigar.
>
> **Acción:** Auditar los 105 casos y registrar el motivo verificable de cada cambio.
>
> **Métrica:** Casos revisados y con motivo documentado / 105.
>
> **Regla aplicada:** Un socio es activo si tiene al menos una actividad activa, incluso si otras figuran inactivas.

**Precisión sobre la respuesta base:** los 63 casos con estados mixtos corresponden a todo 2024. Los datos resumidos enviados al modelo no permiten afirmar cuántos de ellos están entre las 105 transiciones de mayo a junio. La sugerencia de revisarlos es válida como control general, pero no como descripción de esa cohorte.

La rúbrica asigna 1 si cumple el criterio y 0 si no lo cumple. Evalúa solo estas respuestas observadas; no califica la calidad del prompt en general.''')
c('''rubric=pd.DataFrame([
    ['Socios y transiciones correctos',1,1,'Ambas: 257→311; 219→187; 105 y 19 transiciones.'],
    ['Distingue total mensual y cohorte',1,1,'Ambas separan el total de socios de los cambios de estado de los mismos socios.'],
    ['No inventa causas ni asistencia',1,1,'Ambas declaran que falta conocer causa y asistencia.'],
    ['No confunde importes con cobros',1,1,'Ambas aclaran que los importes registrados no prueban cobro efectivo.'],
    ['Aplica la regla confirmada',1,1,'Ambas mencionan activo si tiene al menos una actividad activa.'],
    ['Propone acción y métrica explícita',0,1,'La base sugiere contacto sin indicador; la mejorada define casos con motivo documentado / 105.']
],columns=['criterio','zero_shot','dirigido_one_shot','evidencia'])
display(rubric)
print('Puntaje base:',int(rubric.zero_shot.sum()),'/6 | mejorado:',int(rubric.dirigido_one_shot.sum()),'/6')''')
m('''## 6 · Modelo texto a imagen
La campaña visual se deriva de la **recomendación**, sin pretender que conocemos por qué los socios dejaron de estar activos. Generar la imagen con la herramienta elegida, guardar la imagen y el prompt en el repositorio y evaluar legibilidad, anatomía y adecuación. La consigna permite una herramienta visual externa sin API. No afirmar que su resultado es una campaña probada.''')
c('''image_prompt='Generate a polished, photorealistic square campaign image asset for a gym reactivation concept in Mendoza, Argentina. An adult woman of ordinary athletic ability returning to a gentle workout with a trainer nearby (non-identifiable generated people), natural body proportions, plausible gym equipment, welcoming and nonjudgmental body language. Contemporary modest gym, authentic lived-in details, warm morning light, deep navy and subtle orange color accents. Leave ample clean negative space in upper left for later typography. NO letters, NO text, NO logos, NO brand names, NO before/after transformations, NO claims or numbers. Camera realism, no overprocessed AI sheen. This is a conceptual illustration, not a photo of an actual client.'
print(image_prompt)
# Colab: una vez creada la imagen en la herramienta visual, subir el archivo para documentarla:
# from google.colab import files
# uploaded=files.upload()
# from IPython.display import Image, display
# display(Image(data=next(iter(uploaded.values()))))''')
m('''**Resultado generado (22/09/2026):** [campaña conceptual](assets/campana_reactivacion_gymia.jpg). Revisión visual: sin texto, marcas ni promesas; personas adultas y equipamiento plausibles; espacio libre amplio para diseño. Hay una figura de fondo parcialmente recortada, por lo que la pieza exige revisión antes de publicarse como anuncio. No mide eficacia comercial.\n\n![Campaña conceptual GymIA](assets/campana_reactivacion_gymia.jpg)''')
m('''## 7 · Resultados y conclusiones
En la fuente hay **4.374 filas por actividad**, **4.174 registros socio-mes** y **650 socios únicos** en 2024. En mayo hay 219 activos de 257 registrados; en junio 187 de 311. Entre los mismos socios presentes en ambos meses, 105 cambian de activo a inactivo y 19 en sentido contrario. Hay 63 grupos socio-mes con estados distintos entre actividades y el consolidado aplica la regla «alguna actividad activa».

**Conclusión:** el caso de junio amerita investigar las transiciones antes de ensayar una campaña de reactivación. En las respuestas observadas, ambos prompts respetaron las cifras y límites; el dirigido obtuvo **6/6** frente a **5/6** del base al agregar una métrica explícita y presentar el razonamiento con mayor trazabilidad. Esta diferencia en dos respuestas no demuestra que la técnica siempre mejore el resultado. La imagen es una propuesta visual, no una medición de eficacia comercial. La procedencia sintética y la regla de estado fueron confirmadas por el autor. Para una comparación repetible, registrar el modelo exacto y repetir las pruebas con la misma configuración.''')
n={'cells':C,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
Path('GymIA_Proyecto_Final_Colab.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
print('Notebook written',len(C),'cells')
