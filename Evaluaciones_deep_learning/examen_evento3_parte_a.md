# SI7011 · Evento evaluativo 3 · Parte A

---

## 1. Partir una serie de tiempo

| MAE (°C) | validación | prueba |
|---|---:|---:|
| línea base | 2.44 | 2.62 |
| MLP, partición en el tiempo | 2.50 | 2.67 |
| MLP, partición barajada | 2.30 | 2.57 |

### a) ¿Cuánto mejora en prueba?

 En prueba, el MLP con partición barajada mejora a la línea base solo en 2.62 − 2.57 = **0.05 °C**, no los 0.14 °C que prometía la validación (2.44 − 2.30). Con la partición en el tiempo, la validación habría dicho que el MLP es **peor** que la línea base (2.50 > 2.44, 0.06 °C peor). Eso coincide con lo que pasa en prueba para ese modelo (2.67 > 2.62).

**Explicación:**
- La validación barajada es **optimista**: sobreestima la mejora casi 3 veces (0.14 frente a 0.05).
- La validación en el tiempo es **honesta**: su conclusión ("el MLP no le gana a la persistencia") se mantiene en la prueba.
- Que el MLP barajado dé mejor prueba (2.57) que el temporal (2.67) no reivindica el barajado. Entrenó con el 75 % de la serie en vez del 50 %, es decir, con más datos y más recientes. Lo que falla es su **validación**, que no sirve para decidir.

### b) ¿Por qué barajar produce fuga si ninguna ventana se repite?

 Porque las ventanas se **solapan**: dos ventanas consecutivas comparten 119 de sus 120 horas y tienen objetivos casi iguales. Al barajar, casi toda ventana de validación tiene a su vecina inmediata en entrenamiento, así que validar es interpolar algo casi visto y no predecir un futuro desconocido.

**Explicación:**
- La serie tiene fuerte **autocorrelación**: la temperatura de una hora se parece mucho a la de la hora siguiente.
- "Ninguna ventana repetida" no basta. Lo que importa es que la información de validación no esté en entrenamiento, y aquí lo está casi entera.
- En producción el modelo siempre predice **hacia adelante**, en un periodo que nunca vio (otras estaciones, otro clima). La partición temporal imita esa situación.

### c) ¿Con qué media se normaliza?

 Con la del bloque de entrenamiento, **8.82 °C**, y lo mismo para la desviación estándar. La de toda la serie (9.44 °C) incluye validación y prueba, es decir, información del futuro que el modelo no tendría al desplegarse. Usarla sería otra fuga.

**Explicación:** cualquier estadística que se "aprende" de los datos (medias, desviaciones, escaladores) se calcula solo con entrenamiento y luego se **aplica** igual a validación y prueba. La diferencia 8.82 vs 9.44 muestra que la serie no es estacionaria: el final es más cálido.

---

## 2. El gradiente a lo largo de la ventana

### a) ¿Cuál curva es la RNN simple?

 El **modelo B** (rojo). Su gradiente es ≈ 10⁻³ en la última hora pero cae a ≈ 10⁻²³ hacia las 100 h, unos 20 órdenes de magnitud. A y C se mantienen planos cerca de 10⁻⁶–10⁻⁵ en las 120 h. La forma (línea recta en escala logarítmica, es decir, decaimiento **exponencial**) sale de que el gradiente hacia una hora lejana es un **producto de jacobianas** $\prod_k W_{hh}^\top\,\mathrm{diag}(\tanh'(\cdot))$, una por paso. Si cada factor tiene norma < 1, el producto se encoge geométricamente: es el **gradiente que se desvanece**.

**Explicación:** en una RNN simple, $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$. Para llegar de la pérdida a $x_{t-k}$ hay que multiplicar $k$ veces la misma matriz y la derivada de tanh (≤ 1). Retroceder 100 horas equivale a multiplicar 100 factores menores que 1.

### b) ¿Qué tiene la LSTM que evita esa forma?

 La **celda de memoria** $c_t$, que se actualiza de forma **aditiva**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$. El gradiente a lo largo de la celda es $\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)$, sin multiplicar por una matriz de pesos ni pasar por tanh. Si la compuerta de olvido $f_t \approx 1$, el gradiente pasa casi intacto.

**Explicación:** es la misma idea que el atajo residual de las ResNet: un camino casi identidad por el que la señal fluye sin atenuarse (el *constant error carousel*). La GRU tiene un mecanismo análogo con su compuerta de actualización, $h_t = (1-z_t) h_{t-1} + z_t \tilde h_t$.

### c) Mismo MAE con 120 h y 24 h; ¿por qué la LSTM casi no gana?

 Que la RNN dé 2.33 con 120 h y con 24 h, sumado a la figura (su gradiente más allá de ~24 h es despreciable), dice que la RNN **en la práctica solo usa el último día**, y que para esta tarea eso basta. La LSTM sí recibe gradiente de las 120 horas, pero casi no gana (2.32 vs 2.33) porque **la información útil para predecir la temperatura en 24 h está en las horas recientes**: poder mirar lejos no sirve si allá no hay nada que ayude.

**Explicación:**
- Conservar el gradiente es una condición **necesaria** para aprender dependencias largas, pero no **suficiente** para mejorar: además tiene que haber dependencias largas en los datos.
- Todos los modelos le ganan poco a la persistencia (2.44 → 2.30–2.33), señal de que la tarea es difícil y está dominada por el estado reciente y el ciclo diario.
- La LSTM y la GRU usan 3–4 veces más parámetros (2 065 y 1 553 frente a 529) para una mejora de 0.01–0.03 °C.

---

## 3. Barajar las palabras

| | parámetros | acc. test | acc. test barajado |
|---|---:|---:|---:|
| Modelo X | 1 292 545 | 0.841 | 0.796 |
| Modelo Y | 1 280 193 | 0.854 | 0.854 |
| Modelo Z | 1 313 473 | 0.826 | 0.807 |

### a) ¿Cuál es la bolsa de embeddings?

 El **modelo Y**: su exactitud es idéntica con y sin barajar (0.854 = 0.854), y además es el de menos parámetros (1 280 193, casi todos de la tabla de embeddings). Es exacto y no casualidad porque el **promedio es conmutativo**: $\frac{1}{n}\sum_i e(w_i)$ da el mismo vector en cualquier orden, así que la entrada al clasificador es la misma y cada predicción también.

**Explicación:** no es que "por suerte" acierte el mismo porcentaje. Predice **exactamente lo mismo reseña por reseña** (salvo redondeo de punto flotante). X y Z (CNN 1-D y LSTM) sí miran el orden y por eso cambian al barajar.

### b) ¿Hace falta un modelo que tenga en cuenta el orden?

 Para esta tarea, **poco**. El modelo que ignora el orden es el mejor (0.854 frente a 0.841 y 0.826), y los que lo usan pierden solo 2–5 puntos al barajar (0.841 → 0.796; 0.826 → 0.807). El sentimiento de una reseña se decide sobre todo por **qué palabras aparecen** ("excelente", "aburrida"), no por su orden.

**Explicación:** el orden importa en casos como negaciones ("no es buena"), contrastes ("la actuación es mala pero la historia es buena") o sarcasmo, pero son una minoría. Que X y Z conserven casi todo su acierto con las palabras barajadas muestra que ellos también se apoyan sobre todo en la bolsa de palabras.

### c) GloVe congelado (0.753) vs ajustado (0.859)

 GloVe se entrenó con **co-ocurrencias** en texto general: palabras que aparecen en contextos parecidos quedan cerca, y "bueno" y "malo" aparecen en contextos muy parecidos. Esos vectores no separan bien el **sentimiento**. Al ajustarlos con IMDB, la pérdida de clasificación mueve los vectores para que las palabras positivas y negativas se separen, y para que el vocabulario propio de reseñas de cine adquiera significado útil.

**Explicación:** con los vectores congelados, solo se entrena el clasificador de encima, que debe trabajar con el promedio de vectores pensados para otra cosa. Partir del mismo punto no implica terminar en el mismo: el ajuste **especializa** la representación a la tarea y al dominio.

---

## 4. El cuello de botella del seq2seq

| | parám. | exacta | año (4 cifras) | mes | día |
|---|---:|---:|---|---|---|
| sin atención, d = 24 | 14 204 | 0.004 | 1.00 1.00 0.98 0.95 | 0.76 **0.15** | **0.32 0.09** |
| sin atención, d = 96 | 99 020 | 1.000 | todo 1.00 | | |
| con atención, d = 24 | 16 220 | 1.000 | todo 1.00 | | |

### a) ¿Qué recibe el decoder sin atención?

 Solo el **último estado oculto del encoder**, un único vector de tamaño $d$ que debe resumir toda la frase. Con $d = 24$, 24 números no alcanzan para guardar a la vez día, mes y año (exacta 0.004). Con $d = 96$ sí caben (exacta 1.000).

**Explicación:** ese vector es el **cuello de botella**. Todo lo que el decoder sabe de la entrada tiene que pasar por ahí. Agrandar $d$ lo ensancha, pero cuesta 7 veces más parámetros (99 020 vs 14 204), y con frases más largas volvería a quedarse corto.

### b) ¿Por qué fallan el día y el mes y no el año?

 Porque en la entrada el orden es **día → mes → año** («el primero de mayo de 2 0 2 6»). El año es lo último que lee el encoder y está "fresco" en el estado final. El día y el mes se leyeron al principio y tuvieron que sobrevivir a muchas actualizaciones del estado, que los van sobrescribiendo.

**Explicación:**
- Es el mismo fenómeno de la pregunta 2: la memoria de una red recurrente favorece lo reciente.
- Además, en la salida ISO el día va **al final**, así que el decoder tiene que mantener esa información a lo largo de todos sus pasos de generación antes de escribirla. Las peores posiciones son justo las últimas (0.15 en mes, 0.09 en el último dígito del día).
- Las dos primeras cifras del año casi siempre son "20" (o "19"), así que son fáciles de adivinar (1.00).

### c) ¿Qué cambia con la atención?

 Con atención, en **cada paso** el decoder puede leer **todos** los estados del encoder (uno por palabra de entrada), no solo el último. Calcula pesos $\alpha_{tj} = \mathrm{softmax}_j(\mathrm{score}(s_t, h_j))$ y un contexto $c_t = \sum_j \alpha_{tj} h_j$. Cuando va a escribir el día, mira directamente el estado de «primero». La memoria disponible pasa de $d$ números a $T \times d$, sin agrandar $d$.

**Explicación:** los menos de 2 100 parámetros extra (16 220 − 14 204 = 2 016) solo sirven para **calcular los puntajes** de atención. No agregan capacidad de memoria; lo que cambia es el **acceso** a la información. Por eso con $d = 24$ se resuelve la tarea (1.000) con 6 veces menos parámetros que la versión sin atención y $d = 96$.

---

## 5. Máscaras y posiciones

### a) Asignar P, Q y R


- **P = causal + relleno:** el mapa es **triangular inferior**. «me» solo se atiende a sí mismo (1.00) y todos los pesos por encima de la diagonal son 0; además, las columnas [PAD] son 0.
- **Q = sin máscara:** **ninguna celda es 0**. Las palabras atienden incluso a los [PAD] (p. ej. «banco» → último [PAD] = 0.31, el peso más alto de su fila).
- **R = solo relleno:** las dos columnas [PAD] son 0.00 en todas las filas, pero entre las 5 palabras la atención es completa, también hacia adelante (p. ej. «me» → «el» = 0.29).

**Explicación:** la máscara pone $-\infty$ en los puntajes antes del softmax, así que esos pesos quedan en exactamente 0 y el resto se renormaliza para sumar 1. Detalle que confirma la asignación: la fila de «banco» es igual en P y R (0.17, 0.21, 0.24, 0.15, 0.23), porque «banco» es la última palabra real y con la máscara causal ya puede ver todas las anteriores.

### b) ¿Qué problema produce Q?

 Sin máscara, las palabras reales **reparten atención sobre los [PAD]** y mezclan sus vectores en su representación. Como la cantidad de relleno depende de la frase más larga del lote, **la misma frase da salidas distintas según con qué otras frases se agrupe**. El resultado depende de algo arbitrario y no del contenido.

### c) Permutar sin y con posiciones

 La autoatención sin posiciones es **equivariante a permutaciones**: cada salida es $\sum_j \alpha_{ij} v_j$ con $\alpha_{ij}$ que solo depende del contenido de $q_i$ y $k_j$. Una suma no depende del orden, así que reordenar la entrada solo reordena la salida (la diferencia de $9\cdot10^{-8}$ es redondeo de punto flotante). Con posiciones, cada vector de entrada incluye "dónde está", así que al cambiar el orden cambian $q$, $k$, $v$ y las salidas (0.23). **Consecuencia:** «el perro mordió al hombre» y «el hombre mordió al perro» tienen exactamente las mismas palabras. Sin posiciones el modelo produce el mismo conjunto de salidas para ambas y **no puede distinguir quién mordió a quién**.

**Explicación:** para un Transformer sin posiciones, una frase es un **conjunto** de palabras. Los embeddings de posición son lo único que le da noción de orden.

---

## 6. El Transformer de las fechas y su costo

### a) Sin posición falla en las últimas cifras del año y nunca en el mes

 Sin posición, el modelo ve la entrada como un **conjunto de tokens** y sabe qué hay, pero no en qué orden. El mes viene como una palabra única («mayo»), y su identidad basta para saber el mes, sin importar dónde esté (1.00). El año viene como dígitos sueltos («2 0 2 6»), y los tokens {2, 0, 2, 6} no dicen si es 2026, 2062 o 2602: para saber cuál dígito va tercero y cuál cuarto **hace falta el orden** (0.72 y 0.67). Las dos primeras cifras casi siempre son "20", así que se adivinan sin orden (0.99).

**Explicación:** el modelo con posición resuelve todo (0.999 exacta). El experimento aísla justo lo que aporta la posición: solo importa cuando el significado depende del orden de tokens que, por sí solos, son ambiguos.

### b) Crecimiento de la memoria con T

 Al duplicar $T$, la memoria crece por un factor **4** (0.25 → 1 → 4 → 16 → 64 MB), porque la matriz de atención es $T \times T$ (costo **cuadrático**). Con $T = 4\,096$: 64 × 4 = **256 MB**.

**Explicación:** $4\,096 = 32 \times 128$, así que la memoria es $32^2 = 1\,024$ veces la de $T = 128$: 0.25 × 1 024 = 256 MB. Eso es por capa (y por cabeza y ejemplo del lote, según cómo se mida). Por eso los contextos largos son caros.

### c) Parámetros de $W_Q, W_K, W_V, W_O$ con d = 64

 Cada una es una matriz $64 \times 64$ más un sesgo de 64: $4\,096 + 64 = 4\,160$. Las cuatro suman $4 \times 4\,160 =$ **16 640** parámetros. **No cambia con $T$**: las mismas matrices se aplican a cada token, sea cual sea la longitud.

**Explicación:** $T$ cambia la **memoria y el cómputo** (las activaciones, la matriz $T\times T$ del literal b), pero no el **número de parámetros**. Dividir en varias cabezas tampoco cambia la cuenta: las $h$ cabezas se reparten los 64 canales.

---

## 7. La pérdida del ajuste supervisado

En la figura hay 26 barras: 13 grises/azules (plantilla «Usuario: …» y la pregunta, más la plantilla «Asistente:») y 13 naranjas (respuesta «Protege los derechos básicos de los estadounidenses» + `<fin>`).

### a) ¿Qué barras entran en J y cuál promedio optimiza el SFT?

 Solo las **naranjas** (los tokens de la respuesta y el token de fin). La plantilla y la pregunta se **enmascaran** porque en uso real las escribe el usuario o el sistema y el modelo las recibe como entrada: no queremos enseñarle a generar preguntas, sino a responderlas. El SFT optimiza **2.82**, el promedio sobre los tokens que cuentan.

**Explicación:**
- Las barras enmascaradas siguen entrando como **contexto** (el modelo las lee), pero su pérdida no se suma.
- Comprobación con la figura: las 13 barras naranjas ≈ 10.1 + 2.8 + 2.3 + 0.4 + 5.2 + 0 + 0.6 + 1.4 + 7.7 + 0 + 0 + 0 + 6.4 ≈ 36.9, y 36.9/13 ≈ 2.84 ≈ 2.82 ✔.
- Que `<fin>` esté incluido es clave: así el modelo aprende **cuándo parar** (ver literal c).

### b) Perplejidad

 La perplejidad pasa de $e^{2.06} \approx$ **7.8** a $e^{1.90} \approx$ **6.7**. Significa que, en promedio, en cada token de la respuesta el modelo está tan indeciso como si eligiera al azar entre ~7.8 opciones igual de probables, y después del ajuste entre ~6.7.

**Explicación:** perplejidad = $\exp(J)$ con $J$ en nats por token: es el **número efectivo de opciones**. Hay que verlo frente a un vocabulario de más de 150 000 tokens: ya era muy concentrado, y el ajuste lo reduce ~14 %.

### c) Se detiene en 9 de 12 pero responde «Ballena: aves»

 En 250 pasos aprendió el **formato**: seguir la plantilla de chat, responder corto y emitir el token de fin para **detenerse** (de 2/12 a 9/12). **No** aprendió conocimiento nuevo ni a ser correcto: «Ballena: aves» tiene la forma de una buena respuesta pero es falso (la ballena es un mamífero).

**Explicación:** el SFT corto con LoRA ajusta **comportamiento y estilo**. El conocimiento factual viene del preentrenamiento, y un modelo de 0.5 B tiene poco. Bajar la pérdida premia respuestas **plausibles** en forma, no necesariamente verdaderas.

### d) ¿Se puede comparar J con otro tokenizador?

 **No directamente.** $J$ está en nats **por token**, y otro tokenizador parte el mismo texto en un número distinto de tokens, con vocabularios distintos. Un tokenizador con tokens más largos tiene menos tokens, cada uno más difícil de predecir. Para comparar hay que normalizar por una unidad común: nats o bits **por carácter o por byte**, o la pérdida total sobre el mismo texto.

---

## 8. Elegir cómo usar un modelo preentrenado

| | acc. test | entrenables | tiempo | memoria |
|---|---:|---:|---:|---:|
| LSTM desde cero | 0.778 | 4.2 M | 15 s | 0.7 GB |
| DistilBERT congelado + sonda | 0.801 | 769 | 13 s | 1.0 GB |
| DistilBERT ajustado completo | 0.886 | 67 M | 40 s | 2.2 GB |
| DistilBERT + LoRA + cabeza | 0.891 | 740 k | 34 s | 1.5 GB |

### a) ¿Qué compra la exactitud?

 El **preentrenamiento**. Los ~66 M de pesos congelados de DistilBERT ya aprendieron el idioma con un corpus enorme; LoRA solo los ajusta un poco con sus 740 k parámetros (0.891). La LSTM tiene que aprender todo, embeddings incluidos, desde 2 000 reseñas, y aunque entrena 4.2 M de parámetros, se queda en 0.778.

**Explicación:** lo que cuenta es el **conocimiento que traen los parámetros**, no cuántos se entrenan. Con pocos datos, más parámetros entrenables desde cero suelen significar más sobreajuste.

### b) ¿Por qué la sonda congelada queda tan abajo?

 Porque las representaciones de DistilBERT son **generales** y no se adaptan a la tarea: la sonda (769 = 768 + 1 parámetros) solo puede trazar un hiperplano sobre lo que ya esté linealmente disponible en el vector de [CLS]. Además, DistilBERT se preentrenó solo con *masked language modeling*, así que [CLS] **no fue entrenado como resumen de la frase**. Al ajustar (completo o con LoRA), la atención aprende a reunir en [CLS] lo relevante para el sentimiento (0.801 → 0.886/0.891).

### c) La clínica: 800 notas en español, 3 tareas, GPU de 16 GB

 Elegiría **LoRA**: un solo modelo base congelado y **tres adaptadores pequeños**, uno por tarea (~740 k parámetros cada uno). Con pocos datos iguala o supera al ajuste completo (0.891 vs 0.886), sobreajusta menos, cabe holgado en 16 GB (1.5 GB) y no hay que guardar tres copias de 67 M. Cambiaría:
1. el modelo base por uno en **español**, idealmente clínico o biomédico (DistilBERT es en inglés);
2. la evaluación: con 800 notas, **validación cruzada** estratificada y particionando **por paciente** para evitar fugas, con F1 macro en vez de accuracy si las clases están desbalanceadas.

La compararía contra una **línea base de TF-IDF + regresión logística** (y contra el modelo congelado + sonda).

**Explicación:**
- **Por qué LoRA y no ajuste completo:** con 800 ejemplos, 67 M de parámetros libres sobreajustan fácil. Tres tareas implicarían tres modelos completos, mientras que con LoRA se comparte el base y se cambia solo el adaptador.
- **Por qué no la LSTM:** con 800 notas sería aún peor que con 2 000 reseñas.
- **Por qué TF-IDF + logística:** es barata, fuerte con pocos datos y fácil de interpretar. Si LoRA no le gana con claridad, no vale la pena su complejidad. Es la misma lógica de la "línea base de persistencia" de la pregunta 1.
- **Otros cuidados:** son datos sensibles, así que conviene entrenar localmente o con datos anonimizados. Con 800 ejemplos, reportar la variabilidad entre folds y no un solo número.
