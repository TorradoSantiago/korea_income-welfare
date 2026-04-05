# Original Theoretical Notes (Spanish)

> Extracted from the preserved original notebook so the delivery narrative is readable without opening Jupyter.

<a href="https://colab.research.google.com/github/TorradoSantiago/korea_income-welfare/blob/master/Modelo%20Machine%20Learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#IMPORTACION DE DATOS Y LIBRERIAS

#METADATA

INFORMACION INICIAL

El archivo CSV "Korea_Income_and_Welfare.csv" contiene datos sobre ingresos y bienestar en Corea, con un total de 92,857 registros. Aquí hay un resumen de las columnas disponibles:

id: Identificador único del encuestado.
year: Año de la encuesta.
wave: Número de oleada de la encuesta.
region: Región del encuestado.
income: Ingresos del encuestado.
family_member: Número de miembros de la familia.
gender: Género del encuestado (1 para masculino, 2 para femenino).
year_born: Año de nacimiento del encuestado.
education_level: Nivel de educación del encuestado.
marriage: Estado civil del encuestado.
religion: Si el encuestado sigue alguna religión.
occupation: Ocupación del encuestado.
company_size: Tamaño de la empresa del encuestado.
reason_none_worker: Razón por la cual el encuestado no trabaja.

Aqui podemos ver que tenemos 3 tipos de variables y que no hay valores nulos

###Mapeo de variables
Tenemos que realizar una transformacion del nombre de los datos, ya que eran todos numericos. Las tenemos que poner en categoricas para asi poder sacar mejores insights.

#OBJETIVO ANALITICO

Determinar cómo el nivel de educación afecta los ingresos de una persona en Corea. Este objetivo nos permite cuantificar el impacto de la educación en la capacidad de generación de ingresos, proporcionando insights valiosos para individuos, instituciones educativas, empresas y formuladores de políticas.

**PREGUNTAS E HIPOTESIS**

Hipótesis Nula (H0): El nivel de educación no tiene un efecto significativo en los ingresos de las personas en Corea. Es decir, las diferencias en los ingresos entre distintos niveles educativos pueden atribuirse al azar.
Hipótesis Alternativa (H1): El nivel de educación tiene un efecto significativo en los ingresos de las personas en Corea, donde diferentes niveles educativos conducen a diferentes rangos de ingresos.

**CONTEXTO COMERCIAL**

En un entorno competitivo, tanto individuos como organizaciones buscan maximizar sus potenciales. Para los individuos, entender el retorno de inversión en educación es crucial para tomar decisiones informadas sobre su desarrollo personal y profesional. Para las organizaciones y el sector educativo, comprender la relación entre educación e ingresos ayuda a diseñar programas educativos más efectivos y alineados con las necesidades del mercado laboral.

**PROBLEMA COMERCIAL**

Las decisiones sobre inversión en educación a menudo se realizan sin una comprensión clara de su impacto financiero a largo plazo. Individuos pueden sobreinvertir o subinvertir en su educación, y las instituciones pueden no proporcionar programas que estén alineados con las demandas del mercado laboral. Esto puede llevar a desajustes en la fuerza laboral, desempleo o subempleo, y una economía menos competitiva.

**CONTEXTO ANALITICO**

Desde una perspectiva analítica, abordar este problema implica recolectar y analizar datos sobre ingresos, educación y posiblemente otras variables demográficas y socioeconómicas relevantes. Se utilizarán técnicas de análisis estadístico y aprendizaje automático para examinar la relación entre el nivel educativo y los ingresos, controlando por otras variables que puedan influir en esta relación. El análisis debe ser robusto, controlando por posibles factores de confusión y asegurando que los resultados sean generalizables a la población de Corea.

Este enfoque nos permite no solo responder a preguntas específicas sobre la relación entre educación e ingresos, sino también proporcionar recomendaciones basadas en datos a stakeholders relevantes.

# ANALISIS EXPLORATORIO DE DATOS (EDA)

**Observacion, limpieza e imputacion de nulos**

Ahora nos quedamos sin valores nulos

## VISUALIZACIONES DE DATOS

PASAJE DE MILES DE KWR A USD

**DISTRIBUCION DE INGRESOS EN USD**

La Distribución de los Ingresos en USD muestra una concentración de los ingresos en el extremo inferior, con una cola larga hacia los ingresos más altos, lo que indica que la mayoría de las personas tienen ingresos relativamente bajos, pero hay un pequeño número de individuos con ingresos significativamente altos.

**INGRESOS PROMEDIO POR NIVEL DE EDUCACION**

La visualización de Ingresos Promedio por Nivel de Educación muestra claramente que existe una tendencia de incremento en los ingresos promedio a medida que aumenta el nivel educativo, lo cual apoya la idea de que la educación puede ser un factor importante en la determinación de los ingresos.

**Distribución de Ingresos por Género**

El gráfico de Ingresos por Género muestra la distribución de los ingresos entre los géneros, permitiéndonos observar diferencias en la mediana, rango y outliers entre hombres y mujeres.

**COMPARACION INGRESOS CON MIEMBROS DE FAMILIA**

Este boxplot muestra una comparación clara de la distribución de ingresos entre géneros, con la mediana, cuartiles y rangos intercuartil sin considerar los valores atípicos. Apreciamos que existe una posible disparidad en ingresos con una distribución más amplia para uno de los géneros.

**Ingresos Medianos a lo Largo de los Años**

Este gráfico de líneas representa la mediana de ingresos a través de los años. La tendencia indica cómo han aumentado los ingresos medianos, lo que puede refleja cambios económicos favorables en el país. Con una leve caida del ingreso en el año 2012

**DISTRIBUCION DE EDAD**

Este gráfico te mostraría cómo se distribuyen las edades dentro de tu conjunto de datos. Podrías identificar si tu población es relativamente joven o vieja y observar la presencia de grupos etarios predominantes. Por ejemplo, si hay un pico en el rango de 30-40 años, esto podría indicar una fuerza laboral predominantemente en esa franja de edad.

**INGRESOS POR GENERO Y NIVEL DE EDUCACION**

Este gráfico proporcionaría una comparación visual de los ingresos promedio entre géneros dentro de cada nivel de educación. Podrías descubrir, por ejemplo, si existe una brecha de ingresos significativa entre hombres y mujeres, y si esta brecha varía según el nivel de educación. Esto es fundamental para estudios sobre equidad de género y la influencia de la educación en la igualdad de ingresos.

Distribucion de INGRESOS POR EDAD

La distribución de ingresos por edad muestra una dispersión de ingresos a través de diferentes edades, sin una tendencia clara que indique que ciertas edades tienen consistentemente ingresos más altos o más bajos. La variedad en los niveles de ingresos es considerable en todos los grupos etarios, lo cual sugiere que otros factores además de la edad podrían estar influyendo en los ingresos de las personas.

# MODELO MACHINE LEARNING

### PREPROCESAMIENTO

TRATAMIENTO NULOS

Las columnas "Ocupación", "Tamaño_Empresa", y "Porque_no_trabaja" contienen una cantidad significativa de espacios en blanco. Dado que estamos interesados en construir un modelo de regresión lineal con "Ingresos_USD" como variable objetivo y estas columnas no son numéricas

"Educación" tiene una correlación positiva moderada con "Ingresos_USD" (0.404), sugiriendo que a mayor nivel educativo, podría haber un incremento en los ingresos.
"N_flia" (Número de familiares) también muestra una correlación positiva moderada con "Ingresos_USD" (0.419), lo que podría indicar que los ingresos aumentan con el número de miembros en el hogar, posiblemente reflejando la necesidad de ingresos más altos para sostener familias más grandes.
"Edad" tiene una correlación negativa con "Ingresos_USD" (-0.356), lo que puede reflejar una disminución en los ingresos a medida que las personas envejecen, posiblemente debido a la jubilación o a la transición hacia trabajos menos remunerados.
"Género" muestra una correlación negativa (-0.278) con "Ingresos_USD", lo que podría indicar diferencias de ingresos basadas en el género.

**TRATAMIENTO OUTLIERS CON rango interquartilico**

Elegimos este modelo ya que la distribucion de los datos no es normal por otras razones: Robustez ante Valores Extremos: A diferencia de los métodos basados en la desviación estándar, el IQR es menos sensible a valores extremos. Esto lo hace particularmente útil en distribuciones que no son normales, donde los valores extremos pueden sesgar significativamente la media y la desviación estándar.

Enfoque en la Distribución Media: Al centrarse en la mitad central de los datos (entre el primer y tercer cuartil), el IQR proporciona una medida de variabilidad que refleja la mayoría de la población, ignorando los extremos que podrían no ser representativos de la tendencia general.

Facilidad de Interpretación y Aplicación: El IQR es intuitivo y fácil de calcular, lo que permite identificar y manejar outliers de manera eficiente, algo crucial en etapas tempranas del análisis de datos

### ENTRENAMIENTO

DIVIDIMOS ENTRE X E Y; 30% TEST Y 70% TRAIN**

Este paso prepara nuestros datos para el modelado, separando las variables independientes (X) de la variable dependiente (Y). En la regresión lineal, queremos predecir Y utilizando la información proporcionada por X. Este es un paso crítico para asegurar que nuestro modelo se entrene con las variables correctas.

Entrenamos el modelo con un conjunto de datos y lo probamos con otro conjunto para asegurarnos de que el modelo puede hacer predicciones precisas sobre datos no vistos. Este enfoque ayuda a prevenir el sobreajuste y garantiza que nuestro modelo sea robusto y fiable.

ENTRENAMOS AL MODELO

HACEMOS EL PREDICT X_TEST

REGRESION LINEAL

Regresión Polinomial

RANDOM FOREST

INSIGHTS

Observando los resultados de las predicciones de los tres modelos - Regresión Lineal, Regresión Polinomial, y Random Forest - para el conjunto de prueba, se pueden destacar algunas conclusiones y reflexiones iniciales:

Variedad en las Predicciones:

Las predicciones varían entre los modelos, lo que indica diferencias en cómo cada uno interpreta la relación entre la educación y los ingresos. La Regresión Lineal y la Regresión Polinomial proporcionan estimaciones que, aunque difieren ligeramente en valores, muestran una tendencia similar. En contraste, Random Forest tiende a predecir valores que, para algunos casos, son notablemente más altos o más bajos que los otros modelos.
Sensibilidad a la No Linealidad:

La Regresión Polinomial ajusta las predicciones ligeramente respecto a la Regresión Lineal, lo que sugiere que el modelo capta mejor las complejidades y las relaciones no lineales entre las variables. Sin embargo, la mejora no es drástica, lo que puede indicar limitaciones en la capacidad de los modelos polinomiales de segundo grado para capturar toda la variabilidad en los datos.
Desempeño del Random Forest:

Random Forest muestra variaciones significativas en sus predicciones, lo que puede indicar una mejor capacidad para adaptarse a las peculiaridades de los datos. Este modelo parece identificar mejor la diversidad en los niveles de ingresos, posiblemente debido a su habilidad para manejar interacciones complejas entre variables de manera más eficaz que los modelos lineales y polinomiales.
Implicaciones Prácticas:

Las diferencias en las predicciones entre los modelos subrayan la importancia de seleccionar el enfoque adecuado para el análisis de datos específico. Mientras que modelos más simples como la regresión lineal pueden proporcionar una vista general útil, modelos más complejos como el Random Forest pueden ser necesarios para capturar la totalidad de las dinámicas en juego.
Consideraciones para la Selección de Modelos:

Al elegir entre estos modelos para aplicaciones prácticas, como la formulación de políticas educativas o la planificación de carreras, es crucial considerar no solo la precisión de las predicciones sino también la interpretabilidad del modelo y la relevancia de sus insights. Mientras que Random Forest puede ofrecer predicciones más precisas, la simplicidad de los modelos lineales y polinomiales puede ser ventajosa para comunicar hallazgos y recomendaciones a un público no especializado.
En resumen, estos resultados destacan la complejidad inherente a modelar la relación entre educación e ingresos. La elección del modelo depende tanto del contexto de aplicación como de los objetivos analíticos específicos, equilibrando precisión, interpretabilidad y aplicabilidad de los insights generados.

##ELECCION MODELOS

Mostramos cómo se toman las decisiones basadas en la variable "Educación" para predecir "Ingresos_USD". Los nodos muestran las condiciones o umbrales de división, mientras que las hojas representan los valores de predicción.

### EVALUACION DE LA EFICACIA E INTERPRETACIONES

El análisis mediante diferentes modelos de regresión ha permitido explorar en profundidad la relación entre el nivel de educación y los ingresos en Corea, arrojando luz sobre dinámicas complejas y ofreciendo insights valiosos para individuos, instituciones educativas, empresas y formuladores de políticas. A continuación, se presenta un resumen de los hallazgos y su interpretación en el contexto del objetivo analítico.

Resultados de los Modelos
###REGRESION LINEAL
Con un MAE de 1300.23 USD, RMSE de 1724.17 USD y un R² de 0.33, este modelo presenta una capacidad limitada para explicar la variabilidad de los ingresos, sugiriendo una asociación significativa pero compleja entre las variables de estudio.

###Regresión Polinomial
Presenta una ligera mejora con un MAE de 1299.25 USD, RMSE de 1720.02 USD y un R² de 0.33, indicando que la inclusión de términos polinomiales captura un poco más eficazmente las complejidades de la relación educación-ingresos.

### Random Forest

Muestra el mejor desempeño con un MAE de 1289.11 USD, RMSE de 1713.79 USD y un R² de 0.33, destacando la eficacia de los modelos más complejos y no lineales en capturar las dinámicas subyacentes entre la educación y los ingresos.

#REPORTE DE RESULTADOS

El estudio realizado sobre el dataset, similar a la Encuesta Permanente de Hogares pero enfocado en Corea, ha desentrañado la complejidad de la relación entre el nivel de educación y los ingresos de las personas, un objetivo analítico esencial para entender cómo la educación impacta en la generación de ingresos. Este análisis se apoya en una metodología detallada que abarca desde la preparación de datos hasta el entrenamiento y evaluación de modelos de aprendizaje automático, revelando insights valiosos para una amplia gama de interesados.

A través de la exploración con modelos de regresión lineal, polinomial y Random Forest, se han identificado patrones significativos en la relación educación-ingresos en Corea, destacando unánimemente la educación como un determinante crítico de los ingresos. Este vínculo, marcado por su robustez, se evidencia en las variadas métricas de evaluación empleadas.

El análisis comienza con el modelo de Regresión Lineal, el cual, pese a su simplicidad, marca el punto de partida al confirmar la correlación positiva entre la educación y los ingresos. No obstante, la limitada capacidad de este modelo para abarcar toda la variabilidad sugiere la influencia de factores adicionales y la presencia de dinámicas no lineales.

La Regresión Polinomial avanza en la comprensión de estas dinámicas, mostrando que la relación educación-ingresos no es lineal sino que incluye curvas y puntos de inflexión. Esto nos dice que el impacto de la educación en los ingresos puede ser muy variado, dependiendo de distintos niveles educativos y de la interacción con otras variables.

Sin embargo, es el modelo de Random Forest el que realmente sobresale por su capacidad para capturar con precisión las complejidades de esta relación. La notable mejora en las métricas de evaluación con este modelo señala que la educación interactúa con otros factores de manera sutil y compleja, afectando los ingresos de formas que un enfoque más simplista no podría captar. Este modelo, gracias a su estructura de árboles de decisión múltiples, logra una profundidad de análisis superior, evidenciando cómo distintos factores y su interacción con la educación contribuyen de manera significativa a los patrones de ingresos.

El Random Forest, por lo tanto, emerge no solo como el modelo más preciso sino también como una herramienta poderosa para entender la intrincada relación entre la educación y los ingresos. Su eficacia para manejar la no linealidad y las interacciones múltiples ofrece una ventana a la complejidad del mundo real, haciendo patente que las políticas educativas y las estrategias de desarrollo económico deben considerar un espectro amplio de factores para ser efectivas. Esta visión holística que proporciona el Random Forest invita a los formuladores de políticas, educadores y empresas a adoptar enfoques integrados y multifacéticos para potenciar el impacto de la educación en la generación de ingresos, asegurando así estrategias más inclusivas y efectivas para el bienestar económico en Corea.

###Recomendaciones Basadas en Datos

A la luz de estos hallazgos, es imperativo que las políticas educativas y las estrategias de desarrollo económico consideren la multifacética relación entre la educación y los ingresos. A continuación, se presentan recomendaciones ampliadas para abordar este desafío:

- Fortalecimiento de la Educación Técnica y Vocacional: Más allá de promover la educación superior tradicional, es vital invertir en programas de formación técnica y vocacional. Estos programas deben diseñarse en estrecha colaboración con el sector industrial y tecnológico para garantizar su relevancia y efectividad en equipar a los estudiantes con las habilidades demandadas en la economía actual.

- Desarrollo Profesional Continuo: Fomentar una cultura de aprendizaje continuo entre la fuerza laboral, ofreciendo incentivos para la educación continua y el desarrollo de habilidades a lo largo de la carrera profesional. Las políticas podrían incluir subsidios para la formación profesional, asociaciones entre empresas y universidades, y plataformas en línea para el aprendizaje autónomo.

- Inclusión y Accesibilidad: Asegurar que las oportunidades educativas sean accesibles para todos los segmentos de la sociedad, incluyendo comunidades rurales, minorías étnicas y grupos socioeconómicamente desfavorecidos. Esto puede lograrse mediante la implementación de becas, programas de tutoría, y la ampliación de la infraestructura educativa a regiones menos desarrolladas.

- Bienestar Estudiantil: Reconocer la importancia del bienestar emocional y psicológico de los estudiantes como componente esencial de su éxito educativo. Las iniciativas podrían incluir servicios de consejería, programas de bienestar mental en las escuelas y universidades, y la incorporación de la educación emocional en los currículos.

- Innovación en la Entrega de la Educación: Experimentar con métodos pedagógicos innovadores que aprovechen la tecnología para personalizar el aprendizaje y hacerlo más interactivo y atractivo. Esto incluye la adopción de herramientas digitales, realidad aumentada, y técnicas de gamificación en el proceso educativo.

###Conclusión

Este análisis subraya la educación como un pilar fundamental para el crecimiento personal y el desarrollo económico, ofreciendo una base sólida para estrategias educativas y económicas informadas y efectivas. Los insights generados respaldan la implementación de políticas que no solo aumenten el acceso a la educación sino que también garanticen su calidad y relevancia, contribuyendo al bienestar general de la sociedad en Corea.