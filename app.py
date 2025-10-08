import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Optional, Tuple
import gradio as gr
import os

print(" Iniciando la aplicación del Chatbot CUCEI...")

# --- Configuración ---
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./phi3_hybrid_model/final_adapter" 
CHROMA_DB_PATH = "./chroma_db"
EMBEDDER_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

class ChatCUCEI:
    def __init__(self, base_model, adapter_path, chroma_db_path, embedder_name):
        print("Cargando el sistema ChatCUCEI...")
        
        # --- 1. Cargar el sistema RAG (Base de Datos Vectorial) ---
        print(f"Cargando embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name)
        
        print(f"Conectando a ChromaDB en: {chroma_db_path}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection("profesores_cucei")
        print(f" RAG listo: {self.collection.count()} documentos cargados.")
        
        # --- 2. Cargar el LLM Fine-Tuned ---
        print(f"Cargando modelo base: {base_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # --- Cargar el adaptador LoRA (Fine-Tuning) ---
        if os.path.exists(adapter_path):
            print(f"Cargando adaptador LoRA desde: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("✅ Modelo fine-tuned cargado exitosamente.")
        else:
            print(" Advertencia: No se encontró el adaptador LoRA. Usando el modelo base.")

        self.model.eval()
        print("✅ Modelo listo para inferencia.")
    
    # (Aquí van las funciones de tu clase: normalizar_nombre, buscar_contexto, etc.)
    def normalizar_nombre(self, texto: str) -> str:
        stopwords = {'del', 'de', 'la', 'el', 'profesor', 'profesora', 'profes', 'profe', 'que', 'como', 'es', 'opinas', 'recomiendas', 'sobre', 'a', 'quien','mejor', 'o'}
        texto = re.sub(r'[¿?¡!,.]', '', texto.lower())
        palabras = [p for p in texto.split() if p not in stopwords and len(p) > 1]
        return ' '.join(palabras).upper()

    def buscar_contexto(self, query: str, n_results: int = 15) -> Optional[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results['documents'][0]: return None
        
        nombre_buscado = self.normalizar_nombre(query)
        if not nombre_buscado: return None

        mejor_match = None
        mejor_score = 0.0
        candidatos = {meta['profesor_normalizado']: meta['profesor'] for meta in results['metadatas'][0]}

        for prof_norm in candidatos:
            palabras1 = set(nombre_buscado.split())
            palabras2 = set(prof_norm.split())
            interseccion = len(palabras1 & palabras2)
            if interseccion > mejor_score:
                mejor_score = interseccion
                mejor_match = prof_norm
        
        if mejor_score == 0: return None

        calificaciones, tags, comentarios, nombre_original = [], set(), [], None
        for meta in results['metadatas'][0]:
            if meta['profesor_normalizado'] == mejor_match:
                if not nombre_original: nombre_original = meta['profesor']
                if meta.get('calificacion', 0) > 0: calificaciones.append(meta['calificacion'])
                if meta.get('tags'): tags.update(t.strip() for t in meta['tags'].split(",") if t.strip())
                if meta.get('comentarios'): comentarios.append(meta['comentarios'])
        
        return {
            "profesor": nombre_original,
            "calificacion_promedio": sum(calificaciones) / len(calificaciones) if calificaciones else None,
            "tags": list(tags)[:5],
            "comentarios": sorted(comentarios, key=len, reverse=True)[:2],
        }

    def construir_prompt(self, info: Dict, query: str) -> str:
        contexto = f"Profesor: {info['profesor']}\n"
        if info['calificacion_promedio']: contexto += f"Calificacion: {info['calificacion_promedio']:.1f}/10\n"
        if info['tags']: contexto += f"Tags: {', '.join(info['tags'])}\n"
        if info['comentarios']: contexto += f"Comentario: {info['comentarios'][0][:150]}...\n"
        
        return (
            f"<|system|>Eres ChatCUCEI. Responde brevemente sobre profesores basándote en el contexto.<|end|>\n"
            f"<|user|>Contexto:\n{contexto}\nPregunta: {query}<|end|>\n"
            f"<|assistant|>"
        )

    def generar_respuesta(self, query: str) -> str:
        info = self.buscar_contexto(query)
        if not info:
            return "No encontré información sobre ese profesor. Intenta ser más específico con el nombre."
        
        prompt = self.construir_prompt(info, query)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        
        respuesta_completa = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return respuesta_completa.split("<|assistant|>")[-1].strip()

# --- Instanciar el bot y crear la interfaz ---
try:
    bot = ChatCUCEI(
        base_model=BASE_MODEL_NAME,
        adapter_path=ADAPTER_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        embedder_name=EMBEDDER_NAME
    )

    def chat_function(message, history):
        return bot.generar_respuesta(message)

    iface = gr.ChatInterface(
        fn=chat_function,
        title=" ChatCUCEI - Recomendador de Profesores",
        description="Pregúntame sobre un profesor del CUCEI. Por ejemplo: '¿Qué opinas de Juan Carlos Corona?'",
        examples=[
            "¿Cómo es la profesora Patricia Rosario?",
            "¿Recomiendas a Eloisa Santiago Hernandez?",
            "¿Qué tal es Abel Isai Sanchez Najera?"
        ]
    )
    iface.launch()

except Exception as e:
    print(f" Error fatal al iniciar la aplicación: {e}")
    # Creamos una interfaz de error si algo falla
    with gr.Blocks() as iface:
        gr.Markdown(f"""
        #  Error al Cargar el Modelo
        No se pudo iniciar la aplicación debido a un error:
        `{str(e)}`
        Asegúrate de que todos los archivos (modelo, base de datos vectorial, etc.) se hayan subido correctamente al repositorio de Hugging Face Spaces.
        """)
    iface.launch()