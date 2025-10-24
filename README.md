# Asesor Académico CUCEI

Sistema RAG para consultar evaluaciones de profesores del CUCEI.

## Tecnologías
- ChromaDB 0.5.23
- Groq LLM (Llama 3.3 70B)
- Gradio 4.19.2
- Google Cloud Run

## Demo
🔗 [cucei-advisor-538200693993.us-central1.run.app](https://cucei-advisor-538200693993.us-central1.run.app/)

## Instalación

\\\ash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup_rag_fixed.py
python app.py
\\\

## Autor
Jorge Sánchez - CUCEI
