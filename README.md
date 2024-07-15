# Capstone Project SIC 2023 - Decisionmakers

## Descripción

Este proyecto es el código del modelo del Capstone Project SIC 2023 desarrollado por el equipo Decisionmakers. El objetivo de este modelo es hallar registros similares según unas palabras clave, utilizando técnicas de procesamiento de lenguaje natural (NLP) y el modelo TF-IDF para la vectorización de textos y la similitud del coseno para encontrar los registros más similares.

## Equipo

- **Nombre del equipo:** Decisionmakers

## Estructura del Proyecto

El proyecto se divide en las siguientes secciones:

1. **Carga de Datos:** Uso de la API de datos abiertos para obtener registros públicos.
2. **Preprocesamiento de Datos:** Limpieza y normalización de texto, eliminación de stopwords y lematización utilizando spaCy.
3. **Vectorización:** Uso de TF-IDF para transformar los textos en vectores.
4. **Similitud:** Cálculo de la similitud del coseno para encontrar los registros más similares.
5. **Evaluación:** Evaluación del rendimiento del modelo utilizando métricas como Precision@K, MRR y MAP.

## Requisitos

- Python 3.x
- Librerías: 
  - pandas
  - spacy
  - numpy
  - sklearn
  - sodapy
  - joblib

## Instalación

1. Clona este repositorio:
    ```sh
    git clone https://github.com/tu-repositorio/capstone-project-sic-2023.git
    cd capstone-project-sic-2023
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

3. Descarga el modelo de spaCy para español:
    ```sh
    python -m spacy download es_core_news_sm
    ```

## Uso

### Entrenamiento del Modelo

1. Ejecuta el script principal para cargar y procesar los datos, y entrenar el modelo:
    ```sh
    python main.py
    ```

2. El modelo TF-IDF y el DataFrame procesado se guardarán en archivos `.pkl` para su uso posterior.

## Evaluación del Modelo

1. Ejecuta el script de evaluación para calcular métricas como Precision@K, MRR y MAP:
    ```sh
    python evaluate.py
    ```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, crea una solicitud de extracción (pull request) para discutir los cambios que deseas realizar.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
