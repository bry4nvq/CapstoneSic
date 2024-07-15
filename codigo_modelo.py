# Primera celda:
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import unicodedata
import pandas as pd
!pip install sodapy
from sodapy import Socrata
import time
import joblib

# Instalar y cargar el modelo en español de spaCy
!pip install spacy sodapy
!python -m spacy download es_core_news_sm  # Para lematizar en español
nlp = spacy.load('es_core_news_sm')

# Segunda celda:
# Conectar a la API de datos
client = Socrata("www.datos.gov.co", None)

# Función para hacer una consulta con reintentos
def obtener_datos_con_reintentos(client, dataset_id, query, max_reintentos=5):
    intentos = 0
    while intentos < max_reintentos:
        try:
            results = client.get(dataset_id, where=query, limit=10000)
            return results
        except Exception as e:
            print(f"Error en el intento {intentos + 1}: {e}")
            intentos += 1
            time.sleep(2 ** intentos)  # Espera exponencial entre reintentos
    raise Exception("No se pudo obtener los datos después de varios intentos")

# Consultar los datos publicados en 2024
query = "fecha_de_publicacion >= '2024-01-01T00:00:00.000' AND fecha_de_publicacion < '2025-01-01T00:00:00.000'"
results = obtener_datos_con_reintentos(client, "p6dx-8zbt", query)

# Convertir los resultados a un DataFrame de pandas
df = pd.DataFrame.from_records(results)

# Tercera celda:
columnas_clave = ['entidad', 'departamento_entidad', 'ciudad_entidad', 'ordenentidad',
                  'nombre_del_procedimiento', 'descripci_n_del_procedimiento', 'ciudad_de_la_unidad_de',
                  'tipo_de_contrato'
                  ] ## columnas que aportan información reelevante

# Cuarta celda:
# Función para limpiar el texto
def limpiar_texto(text):
    if isinstance(text, str):
        text = text.lower()  # Convertir a minúsculas
        text = re.sub(r'\d+', '', text)  # Eliminar dígitos
        text = re.sub(r'\W+', ' ', text)  # Reemplazar caracteres no alfanuméricos por un espacio
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # Eliminar caracteres no ASCII
        text = text.strip()  # Eliminar espacios en blanco al inicio y al final
        text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios por uno solo
    return text

# Quinta celda:
# Aplicar la limpieza a las columnas clave
for columna in columnas_clave:
    df[columna] = df[columna].apply(limpiar_texto)

# Sexta celda:
# Crear una columna combinada de palabras clave
df['Palabras Clave'] = df[columnas_clave].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Séptima celda:
# Preprocesar los textos
def preprocess_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    lemmatized_tokens = [token for token in lemmatized_tokens if len(token) > 2]  # Eliminar tokens muy cortos

    # Eliminar palabras específicas no deseadas
    palabras_no_deseadas = {'cuantiar', 'cuantiir', 'inter'}
    lemmatized_tokens = [token for token in lemmatized_tokens if token not in palabras_no_deseadas]

    return ' '.join(lemmatized_tokens)

# Procesar la columna palabras clave
df['processed_palabras_clave'] = df['Palabras Clave'].apply(preprocess_text)

# Octava celda:
# Inicializar el vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Entrenar el modelo y añadirlo a la variable X
X = vectorizer.fit_transform(df['processed_palabras_clave'])

# Novena celda:
# Definir una función para encontrar los registros más similares con la búsqueda
def registros_mas_similares(input_text, df, vectorizer, X, top_n=5):
    df_1 = df.copy()

    if isinstance(input_text, list):
        input_text = ' '.join(input_text)

    # Preprocesar el texto del usuario
    processed_input = preprocess_text(input_text)

    # Vectorizar el texto del usuario
    input_vector = vectorizer.transform([processed_input])

    # Ejecutar la similitud del coseno
    similarities = cosine_similarity(input_vector, X)

    # Obtener las filas más similares
    top_indices = similarities.argsort()[0][-top_n:][::-1]

    # Devolver las filas correspondientes del DataFrame
    return df_1.iloc[top_indices, [2, 6, 10]]

# Décima celda:
# Usar el modelo
input_text = ["salud", 'Huila', "casa"]
registros_mas_similares = registros_mas_similares(input_text, df, vectorizer, X, 7)
print(registros_mas_similares)

# Onceava celda:
# Para un primer acercamiento en el análisis de tendencias, realizamos gráfico de barras para ver las palabras más frecuentes
# y una nube de palabras para visualizar lo mismo de una mejor manera
all_words = ' '.join(df['processed_palabras_clave'])
word_freq = Counter(all_words.split())
most_common_words = dict(word_freq.most_common(20))
# Guardar el modelo TF-IDF
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Guardar el array TF-IDF
joblib.dump(X, 'tfidf_matrix.pkl')

# Guardar el DataFrame procesado
df.to_pickle('processed_dataframe.pkl')

plt.figure(figsize=(10, 6))
plt.bar(most_common_words.keys(), most_common_words.values())
plt.title('Frecuencia de Palabras Clave')
plt.xticks(rotation=45)
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_words)  # acá podemos ajustar los hiperparámetros si queremos jugar con la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
