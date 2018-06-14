Clasificación de emociones en mensajes de tuits

Consiste en proporcionarle como datos de entrada varias y distintas opiniones extraídas de Twitter en forma de texto, donde la red pasará por una etapa de aprendizaje y tras un número elevado de opiniones distintas y de los distintos tipos que se le proporcionen (enojo, diversión, miedo, tristeza); modificará sus valores o pesos en sus conexiones por medio de un algoritmo, así este entrenamiento continuará hasta que la red sea capaz de que al pasarle cualquier opinión, esta la clasifique según lo que por sí sola haya aprendido de los datos que al principio le proporcionamos.

Requisitos

Tamaño de texto
140 - 280 caracteres por cada mensaje de tuit.
Lenguaje de programación y versión
Python 3.6 o mayor
Librerías básicas
1.- nltk (Natural Language Tool Kit): NLTK es una plataforma que trabaja con datos de lenguaje humano. Proporciona interfaces fáciles de usar a más de 50 recursos corporales y léxicos como WordNet, junto con un conjunto de bibliotecas de procesamiento de texto para clasificación, tokenización, derivación, etiquetado, análisis y razonamiento semántico, envoltorios para bibliotecas de procesamiento de lenguaje natural de fuerza industrial, y un foro de discusión activo.
2.- spaCy: librería usada para etiquetado, análisis sintáctico y reconocimiento de entidades. Los modelos se han diseñado e implementado desde cero específicamente para spaCy, para brindarle un equilibrio inigualable de velocidad, tamaño y precisión. 
3.- seaborn / matplotlib: librerías que dan un alto nivel para poder realizar gráficas mediante datos estadísticos.
4.- keras: API de redes neuronales de alto nivel, capaz de ejecutarse sobre TensorFlow, CNTK o Theano (librerías destinadas al cálculo numérico computacional). Fue desarrollado con un enfoque en permitir la experimentación rápida acerca de distintos modelos de redes neuronales artificiales.

Redes Neuronales Convolucionales
Una red neuronal convolucional son similares a redes multicapa, pero su principal ventaja es que cada parte de la red se le entrena para realizar una tarea reduciendo significativamente el número de capas ocultas, por lo que el entrenamiento se vuelve más rápido. 
Se usan principalmente para:
-	Detección / Categorización de objetos
-	Clasificación de escenarios / imágenes
-	Series o tiempo de señales de audio
-	Clasificación de texto
-	Análisis de sentimientos

Ejecución
python3 Ini_Red.py
