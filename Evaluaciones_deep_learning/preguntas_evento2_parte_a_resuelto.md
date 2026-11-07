# SI7011 · Evento evaluativo 2 · Parte A

---

## 1. Permutar los píxeles

Datos de la figura (CIFAR-10, 10 épocas):

| | parámetros | orden original | permutados |
|---|---:|---:|---:|
| Modelo X | 1 707 274 | 0.441 | 0.438 |
| Modelo Y | 66 570 | 0.604 | 0.405 |

### a) ¿Cuál es el MLP y cuál la CNN?

 X es el MLP y Y es la CNN. Con la permutación, X casi no cambia (0.441 → 0.438) y Y cae mucho (0.604 → 0.405); además X tiene 1 707 274 parámetros frente a 66 570 de Y, lo típico de una capa densa sobre 3 072 entradas frente a filtros compartidos.

**Explicación:**
- El número de parámetros de X cuadra exactamente con un MLP 3072 → 512 → 256 → 10:
  - 3 072·512 + 512 = 1 573 376
  - 512·256 + 256 = 131 328
  - 256·10 + 10 = 2 570
  - **Total = 1 707 274** ✔
- Una CNN pequeña tiene pocos parámetros porque cada filtro (p. ej. 3×3) se reutiliza en toda la imagen.

### b) ¿Por qué la permutación afecta a uno y no al otro?

 Al MLP no le importa el orden: una permutación fija solo reordena las filas de su primera matriz de pesos, así que puede aprender exactamente la misma función. La CNN supone **localidad / estructura espacial** (los píxeles vecinos están relacionados y los patrones son locales y se repiten en distintas posiciones); al barajar, los "vecinos" que ve un filtro 3×3 ya no son vecinos reales y esa suposición se rompe.

**Explicación:**
- Para el MLP, cada píxel de entrada tiene su propio peso independiente; la red nunca "sabe" qué píxel está al lado de cuál. Permutar = renombrar entradas. La diferencia 0.441 vs 0.438 es ruido (inicialización, orden de los minibatches).
- La CNN tiene un **sesgo inductivo** de localidad y equivarianza a traslaciones. Ese sesgo es justamente lo que le permite ganarle al MLP con 25 veces menos parámetros (0.604 vs 0.441). Al permutar, el sesgo deja de ser cierto y queda incluso por debajo del MLP (0.405 < 0.438).
- Sigue por encima del azar (0.10) porque aún puede aprovechar estadísticas globales como el color promedio.

---

## 2. Formas, parámetros y campo receptivo

Entrada: 3 × 64 × 64.
Fórmula de salida: `H_out = floor((H + 2·padding − kernel) / stride) + 1`.

### a) Formas



| capa | salida C × H × W |
|---|---|
| Conv2d(3, 32, k=3, p=1) | **32 × 64 × 64** |
| Conv2d(32, 64, k=3, p=1, s=2) | **64 × 32 × 32** |
| MaxPool2d(2) | **64 × 16 × 16** |

**Explicación:**
- Conv 1: (64 + 2 − 3)/1 + 1 = 64 → `padding=1` con kernel 3 conserva el tamaño ("same").
- Conv 2: floor((64 + 2 − 3)/2) + 1 = floor(31.5) + 1 = 32 → el stride 2 divide a la mitad.
- MaxPool2d(2): kernel 2, stride 2 (por defecto) → 32/2 = 16.
- Las ReLU no cambian la forma. El número de canales de salida es siempre el `out_channels` de la conv.

### b) Parámetros

 La segunda convolución tiene 32·64·3·3 + 64 = **18 496** parámetros. Una capa densa de 32·64·64 = 131 072 entradas a 64·32·32 = 65 536 salidas tendría 131 072·65 536 + 65 536 = **8 590 000 128** (≈ 8.6 mil millones). La diferencia la explican la **conectividad local** (cada salida ve solo una ventana 3×3) y el **compartir pesos** (el mismo filtro se usa en todas las posiciones).

**Explicación:**
- Conv: cada uno de los 64 filtros tiene 32 × 3 × 3 = 288 pesos + 1 sesgo → 64 · 289 = 18 496.
- Densa: cada salida se conecta con todas las entradas → (entradas × salidas) + sesgos.
- La razón es ≈ 465 000 veces más parámetros. El número de parámetros de la conv **no depende del tamaño de la imagen**; el de la densa crece con H·W de entrada y de salida.

### c) Campo receptivo

 A la salida de la segunda convolución cada posición ve una región de **5 × 5** píxeles de la entrada (en los 3 canales); a la salida del MaxPool, **7 × 7**.

**Explicación:** se acumula con `RF_nuevo = RF_anterior + (k − 1) · salto`, donde el salto ("jump") es el producto de los strides anteriores.

| capa | k | stride | salto antes | campo receptivo |
|---|---|---|---|---|
| entrada | – | – | 1 | 1 |
| Conv 1 | 3 | 1 | 1 | 1 + 2·1 = **3** |
| Conv 2 | 3 | 2 | 1 | 3 + 2·1 = **5** (salto pasa a 2) |
| MaxPool | 2 | 2 | 2 | 5 + 1·2 = **7** (salto pasa a 4) |

(El stride de la conv 2 no agranda su propio campo receptivo; agranda el de las capas que vienen después, por eso el pool suma 2 en vez de 1.)

---

## 3. BatchNorm y data augmentation

Datos (CIFAR-10, misma red VGG, época 12):

| corrida | entrenamiento | validación | brecha (train − val) |
|---|---:|---:|---:|
| A | 0.688 | 0.656 | 0.032 |
| B | 0.502 | 0.501 | 0.001 |
| C | 0.800 | 0.611 | 0.189 |

### a) Asignar A, B y C


- **B = sin BatchNorm:** es la que peor entrena (0.502 en entrenamiento); sin BN la optimización es más lenta y en 12 épocas no alcanza a ajustar.
- **C = con BatchNorm (sin augmentation):** es la que más ajusta el entrenamiento (0.800) pero con la mayor brecha (0.800 − 0.611 = 0.189): sobreajusta.
- **A = BatchNorm + augmentation:** la mejor en validación (0.656) con brecha pequeña (0.032).

**Explicación:**
- BatchNorm normaliza las activaciones de cada capa, lo que estabiliza los gradientes y permite aprender más rápido: por eso las dos corridas con BN (A y C) superan a B en entrenamiento.
- B tiene train ≈ val (0.502 / 0.501): no sobreajusta, simplemente **subajusta** porque aprende lento.
- La augmentation actúa como regularizador: baja el accuracy de entrenamiento y sube el de validación.

### b) Menos accuracy de entrenamiento pero más en validación

 A tiene menos accuracy de entrenamiento que C (0.688 < 0.800) pero más en validación (0.656 > 0.611) porque la augmentation hace que el modelo vea en cada época versiones distintas (recortes, flips) de las imágenes: no puede memorizarlas y aprende rasgos que generalizan. Calcularía la **brecha de generalización train − val**: 0.032 en A frente a 0.189 en C.

**Explicación:**
- El accuracy de entrenamiento de A se mide sobre imágenes aumentadas, que son una tarea más difícil y cambiante; por eso es más bajo.
- C, sin augmentation, ve exactamente las mismas 50 000 imágenes cada época y las va memorizando (train sube, val se queda atrás).
- Lo que importa es el desempeño en datos nuevos, y ahí A gana. Una brecha grande es la firma del sobreajuste.

---

## 4. Profundidad y atajos

| | parámetros | pérdida train | acc. train | acc. val | brecha |
|---|---:|---:|---:|---:|---:|
| Red P | 269 722 | 1.189 | 0.565 | 0.529 | 0.036 |
| Red Q | 272 474 | 0.992 | 0.639 | 0.606 | 0.033 |

### a) «La red P sobreajusta»

 No estoy de acuerdo. P tiene **peor** ajuste en entrenamiento que Q (accuracy 0.565 vs 0.639; pérdida 1.189 vs 0.992), y su brecha train − val es pequeña (0.565 − 0.529 = 0.036, casi igual a la de Q). Eso es un problema de **optimización / subajuste**, no de sobreajuste.

**Explicación:**
- Sobreajustar = ajustar muy bien el entrenamiento y mal la validación (brecha grande, como la corrida C de la pregunta 3).
- Aquí P ni siquiera logra ajustar bien el entrenamiento. Es el **problema de degradación**: una red profunda sin atajos entrena peor que una igual con atajos, aunque tenga la misma capacidad.

### b) ¿Cuál tiene atajos?

 **Q** tiene atajos (menor pérdida de entrenamiento, 0.992 vs 1.189). La salida de un bloque residual es

$$a_{l+1} = \mathrm{ReLU}\big(F(a_l) + a_l\big)$$

(o simplemente $a_{l+1} = a_l + F(a_l)$), donde $F$ son las convoluciones del bloque. El atajo ayuda porque el gradiente tiene un camino directo: $\frac{\partial a_{l+1}}{\partial a_l} = I + \frac{\partial F}{\partial a_l}$, así que no se desvanece al atravesar muchas capas, y al bloque le basta con aprender $F \approx 0$ para comportarse como la identidad.

**Explicación:**
- Sin atajos, el gradiente que llega a las primeras capas es un producto de muchas jacobianas; si son "pequeñas" se desvanece y esas capas casi no aprenden.
- Con atajos, el término $I$ garantiza que la señal (hacia adelante) y el gradiente (hacia atrás) pasen intactos aunque $F$ todavía no haya aprendido nada útil.
- El bloque aprende el **residuo** (lo que hay que corregir sobre la entrada) en vez de toda la transformación; aprender "no cambiar nada" es trivial.
- Por eso la ventaja se ve en la **pérdida de entrenamiento**: es una mejora de optimización.

### c) ¿De dónde salen los 2 752 parámetros extra?

 De los atajos de **proyección** en los bloques donde cambia la forma (se duplican los canales y se reduce la resolución con stride 2). Ahí no se puede sumar la identidad porque $a_l$ y $F(a_l)$ tienen formas distintas, y se usa una convolución 1×1 (con stride 2) + BatchNorm.

**Explicación (cuenta exacta, ResNet-20 de CIFAR con etapas de 16 → 32 → 64 canales):**

| transición | conv 1×1 (sin sesgo) | BatchNorm (γ, β) | total |
|---|---:|---:|---:|
| 16 → 32 | 16·32 = 512 | 2·32 = 64 | 576 |
| 32 → 64 | 32·64 = 2 048 | 2·64 = 128 | 2 176 |
| **total** | | | **2 752** ✔ |

272 474 − 269 722 = 2 752: cuadra exactamente.

---

## 5. Tres estrategias de transfer learning

Datos (Oxford-IIIT Pets, 37 razas; azar = 1/37 ≈ 0.027):

| imágenes por raza | I | II | III |
|---|---:|---:|---:|
| 5 | 0.588 | 0.086 | 0.762 |
| 20 | 0.789 | 0.174 | 0.834 |
| ≈ 80 (todas) | 0.872 | 0.305 | 0.849 |

### a) Asignar I, II y III


- **II = desde cero:** la peor en todos los regímenes; con 5 imágenes por raza apenas llega a 0.086 (cerca del azar, 0.027) y con todas solo a 0.305. Sin preentrenamiento, 11 M de parámetros no se aprenden con unas pocas miles de imágenes.
- **III = linear probe (backbone congelado):** la mejor con pocos datos (0.762 con 5 por raza) y casi plana (solo llega a 0.849): solo se entrena una capa lineal sobre features de ImageNet, que ya son buenas pero no se adaptan.
- **I = fine-tuning completo:** arranca por debajo del probe (0.588) pero sube más y es la mejor con todos los datos (0.872).

**Explicación:** ImageNet contiene muchos perros y gatos, así que sus features ya separan razas bastante bien → el linear probe funciona desde el principio. Entrenar desde cero no aprovecha nada de eso.

### b) Con 5 imágenes I < III, con todas I > III

 Con 5 imágenes por raza (≈185 en total), el fine-tuning ajusta ~11 M de parámetros con muy pocos datos: sobreajusta y desordena las features preentrenadas (0.588), mientras que el linear probe solo entrena ≈19 000 parámetros (512·37 + 37) sobre features fijas y generaliza mejor (0.762). Con todas las imágenes hay datos suficientes para que el fine-tuning **adapte** el backbone a rasgos finos de las razas, cosa que el probe no puede hacer porque está congelado (0.872 vs 0.849).

**Explicación:** es el compromiso sesgo–varianza:
- Linear probe = mucho sesgo (features fijas), poca varianza → gana con pocos datos, pero tiene techo.
- Fine-tuning = más flexibilidad, más varianza → necesita más datos, pero su techo es más alto.

### c) ¿Por qué η distinto para backbone (10⁻⁴) y cabeza (10⁻³)?

 El backbone ya está cerca de una buena solución y solo necesita ajustes pequeños; un η grande destruiría las features preentrenadas (olvido catastrófico), sobre todo al principio, cuando la cabeza es aleatoria y manda gradientes grandes y ruidosos hacia atrás. La cabeza nueva parte de cero y necesita un η mayor para aprender rápido.

---

## 6. CNN frente a ViT

| modelo | parámetros | acc. val | tiempo |
|---|---:|---:|---:|
| CNN (VGG + BN) | 72 954 | 0.693 | 21 s |
| ViT pequeño (patches 4×4, d=128, 6 bloques) | 811 146 | 0.603 | 75 s |
| ViT sin embeddings de posición | 811 146 | 0.551 | 78 s |
| ViT-B/16 preentrenado + linear probe | ≈ 86 M | 0.943 | — |

### a) Más parámetros pero pierde; el preentrenado gana

 La CNN trae incorporados sesgos inductivos (localidad, compartir pesos, equivarianza a traslaciones) que encajan con las imágenes, así que aprende bien con 50 000 imágenes y pocos parámetros (0.693 con 72 954). El ViT no trae esos sesgos: debe aprender desde los datos qué patches se relacionan, y 50 000 imágenes no alcanzan (0.603 con 811 146). El ViT-B/16 preentrenado ya aprendió esa estructura con millones de imágenes de ImageNet; sus features son tan buenas que basta una capa lineal encima para llegar a 0.943.

**Explicación:**
- Más parámetros no implica mejor si no hay datos suficientes para aprovecharlos: el ViT además es 3.5 veces más lento (75 s vs 21 s) para un peor resultado.
- La lección es: **sesgo inductivo vs datos**. Con pocos datos ganan los sesgos (CNN); con muchos datos, el modelo flexible (ViT) los aprende y los supera.
- Ojo: la comparación no es justa en datos (el ViT-B/16 vio ImageNet y es ~1 000 veces más grande que la CNN); muestra lo que aporta el preentrenamiento a escala.

### b) Sin embeddings de posición

 Pierde **dónde** está cada patch: la autoatención es equivariante a permutaciones, así que el modelo ve la imagen como una "bolsa de patches" sin saber su arreglo espacial. Aun así queda muy por encima del azar (0.551 vs 0.10) porque dentro de cada patch de 4×4 los píxeles conservan su orden, así que siguen disponibles colores, texturas y bordes locales, y saber **qué** patches hay ya es muy informativo (p. ej. muchos patches de cielo azul + metal sugieren un avión).

**Explicación:** la caída es solo de 0.603 a 0.551. Es la misma idea de la pregunta 1, pero al revés: aquí se baraja el orden **entre** patches, no **dentro** de ellos, y la información local se mantiene.

### c) Número de tokens y crecimiento de la atención


- 32×32 con patches 4×4: (32/4)² = 8·8 = 64 patches + [CLS] = **65 tokens**.
- 64×64 con patches 4×4: (64/4)² = 16·16 = 256 patches + [CLS] = **257 tokens**.
- La matriz de atención pasa de 65² = 4 225 a 257² = 66 049 entradas: crece por un factor de **≈ 15.6 (≈ 16)**, porque el costo es **cuadrático** en el número de tokens (4 veces más tokens → ~16 veces más entradas), por cabeza y por bloque.
