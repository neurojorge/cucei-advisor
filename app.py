# -*- coding: utf-8 -*-
import chromadb
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import gradio as gr
from groq import Groq
import csv
import os
from datetime import datetime
import hashlib

print("="*70)
print("CUCEI-ADVISOR (v4.0 - Groq Edition)")
print("="*70)

# Inicializar Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not configured")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Cache para consultas repetidas
RESPONSE_CACHE = {}
CACHE_MAX_SIZE = 100

def cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

class ChatCUCEI:
    def __init__(self, chroma_db_path="chroma_db"):
        print("Inicializando ChatCUCEI con Groq...")
        self.chroma_db_path = chroma_db_path
        self._chroma_client = None
        self._collection = None
        
        # Verificar que existe el directorio de ChromaDB
        if not os.path.exists(self.chroma_db_path):
            print(f"ERROR: No se encuentra el directorio {self.chroma_db_path}")
            raise FileNotFoundError(f"Directorio {self.chroma_db_path} no encontrado")
        
        print("ChatCUCEI inicializado")
    
    @property
    def chroma_client(self):
        if self._chroma_client is None:
            print(f"Conectando a ChromaDB en: {self.chroma_db_path}")
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=self.chroma_db_path,
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
                print("✓ Conexión exitosa a ChromaDB")
            except Exception as e:
                print(f"✗ Error al conectar con ChromaDB: {e}")
                raise
        return self._chroma_client
    
    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.chroma_client.get_collection("profesores_cucei")
                count = self._collection.count()
                print(f"ChromaDB cargada: {count} documentos encontrados")
                
                if count == 0:
                    print("WARNING: La coleccion esta vacia")
                    
            except Exception as e:
                print(f"✗ Error al cargar colección: {e}")
                print("Intentando crear nueva colección...")
                try:
                    self._collection = self.chroma_client.create_collection(
                        "profesores_cucei",
                        metadata={"hnsw:space": "cosine"}
                    )
                    print("Nueva coleccion creada")
                except Exception as create_error:
                    print(f"Error al crear coleccion: {create_error}")
                    raise
        return self._collection
    
    def normalizar_nombre(self, texto: str) -> str:
        
        stopwords = {'del', 'de', 'la', 'el', 'profesor', 'profesora', 'profes', 'profe',
                     'que', 'como', 'es', 'opinas', 'recomiendas', 'sobre', 'a', 'quien',
                     'mejor', 'o', 'dr', 'dra', 'mtro', 'mtra', 'ing'}
        texto = re.sub(r'[¿?¡!,.]', '', texto.lower())
        palabras = [p for p in texto.split() if p not in stopwords and len(p) > 1]
        return ' '.join(palabras).upper()
    
    def extraer_nombres(self, query: str) -> List[str]:
        
        query_limpia = re.sub(r'[¿?¡!,.]', '', query.lower())
        if ' o ' in query_limpia:
            partes = query_limpia.split(' o ')
            return [self.normalizar_nombre(p) for p in partes if self.normalizar_nombre(p)]
        return [self.normalizar_nombre(query)]
    
    def similitud_rapida(self, nombre1: str, nombre2: str) -> float:
        """Calcula similitud entre dos nombres"""
        palabras1 = set(nombre1.split())
        palabras2 = set(nombre2.split())
        if not palabras1 or not palabras2: 
            return 0.0
        interseccion = len(palabras1 & palabras2)
        union = len(palabras1 | palabras2)
        return interseccion / union if union > 0 else 0.0
    
    def buscar_mejor_match(self, nombre_query: str, candidatos: List[Dict]) -> Tuple[Optional[str], float]:
        """Encuentra el mejor match de profesor en los candidatos"""
        if not nombre_query: 
            return None, 0.0
        
        mejor_profesor, mejor_score = None, 0.0
        profesores_vistos = set()
        
        for meta in candidatos:
            prof_norm = meta.get('profesor_normalizado', '')
            if not prof_norm or prof_norm in profesores_vistos: 
                continue
            profesores_vistos.add(prof_norm)
            
            score = self.similitud_rapida(nombre_query, prof_norm)
            if score > mejor_score:
                mejor_score, mejor_profesor = score, prof_norm
        
        if mejor_score < 0.3: 
            return None, mejor_score
        return mejor_profesor, mejor_score
    
    def buscar_contexto(self, query: str, n_results: int = 25) -> Optional[Dict]:
        """Busca contexto de un profesor en la base de datos"""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            
            if not results or not results.get('documents') or not results['documents'][0]: 
                print(f"No se encontraron resultados para: {query}")
                return None
            
            nombres = self.extraer_nombres(query)
            mejor_match, score = self.buscar_mejor_match(nombres[0], results['metadatas'][0])
            
            if not mejor_match:
                print(f"No se encontró match suficiente (score: {score:.2f})")
                return None
            
            print(f"Match encontrado: {mejor_match} (score: {score:.2f})")
            
            calificaciones, tags, comentarios, nombre_original = [], set(), [], None
            
            for meta in results['metadatas'][0]:
                if meta.get('profesor_normalizado') == mejor_match:
                    if not nombre_original: 
                        nombre_original = meta.get('profesor', '')
                    
                    cal = meta.get('calificacion', 0)
                    if cal and cal > 0: 
                        calificaciones.append(cal)
                    
                    if meta.get('tags'):
                        tags.update(t.strip() for t in str(meta['tags']).split(",") if t.strip())
                    
                    if meta.get('comentarios'):
                        comentarios.append(str(meta['comentarios']))
            
            comentarios_ordenados = sorted(comentarios, key=len, reverse=True)
            
            return {
                "profesor": nombre_original,
                "calificacion_promedio": sum(calificaciones) / len(calificaciones) if calificaciones else None,
                "tags": list(tags)[:5],
                "comentarios": comentarios_ordenados[:3],
                "num_evaluaciones": len(calificaciones)
            }
            
        except Exception as e:
            print(f"Error en busqueda de contexto: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def buscar_comparacion(self, query: str) -> Optional[Dict]:
        """Busca información para comparar dos profesores"""
        nombres = self.extraer_nombres(query)
        if len(nombres) < 2:
            return None
        
        resultados = {}
        
        for nombre in nombres[:2]:
            try:
                results = self.collection.query(query_texts=[nombre], n_results=40)
                if not results or not results.get('documents') or not results['documents'][0]:
                    continue
                
                mejor_match, score = self.buscar_mejor_match(nombre, results['metadatas'][0])
                if not mejor_match or score < 0.3:
                    continue
                
                tags, comentarios = {}, []
                nombre_original = None
                
                for meta in results['metadatas'][0]:
                    if meta.get('profesor_normalizado') == mejor_match:
                        if not nombre_original:
                            nombre_original = meta.get('profesor', '')
                        
                        if meta.get('tags'):
                            for tag in str(meta['tags']).split(","):
                                tag = tag.strip()
                                if tag:
                                    tags[tag] = tags.get(tag, 0) + 1
                        
                        if meta.get('comentarios'):
                            comentarios.append(str(meta['comentarios']))
                
                if nombre_original and comentarios:
                    tags_ordenados = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                    comentarios_ordenados = sorted(comentarios, key=len, reverse=True)
                    
                    # Seleccionar mejores comentarios
                    comentarios_seleccionados = []
                    comentarios_seleccionados.extend([c for c in comentarios_ordenados if len(c) > 100][:3])
                    comentarios_seleccionados.extend([c for c in comentarios_ordenados if 50 <= len(c) <= 100][:2])
                    comentarios_seleccionados.extend([c for c in comentarios_ordenados if len(c) < 50][:2])
                    comentarios_unicos = list(dict.fromkeys(comentarios_seleccionados))[:7]
                    
                    resultados[nombre_original] = {
                        "tags": tags_ordenados[:8],
                        "num_comentarios": len(comentarios),
                        "comentarios_muestra": comentarios_unicos
                    }
                    
            except Exception as e:
                print(f"Error procesando profesor {nombre}: {e}")
                continue
        
        return resultados if len(resultados) >= 2 else None
    
    def generar_respuesta_groq(self, prompt: str, max_tokens: int = 300) -> str:
        """Genera respuesta usando Groq API"""
        if not groq_client:
            return "Error: Groq API no configurada. Por favor configura GROQ_API_KEY."
        
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un asistente académico del CUCEI. Responde de forma concisa y profesional basándote SOLO en la información proporcionada."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error con Groq API: {e}")
            return f"Error al generar respuesta: {str(e)}"
    
    def generar_comparacion_detallada(self, comparacion: Dict) -> str:
        """Genera comparación detallada entre profesores"""
        if not comparacion or len(comparacion) < 2:
            return None
        
        profesores = list(comparacion.keys())
        prof1, prof2 = profesores[0], profesores[1]
        info1, info2 = comparacion[prof1], comparacion[prof2]
        
        if not info1.get('comentarios_muestra') or not info2.get('comentarios_muestra'):
            return None
        
        comentarios1 = '\n'.join(['• ' + com for com in info1['comentarios_muestra'][:5]])
        comentarios2 = '\n'.join(['• ' + com for com in info2['comentarios_muestra'][:5]])
        tags1_str = ', '.join([f'"{tag}" ({freq}x)' for tag, freq in info1['tags'][:6]])
        tags2_str = ', '.join([f'"{tag}" ({freq}x)' for tag, freq in info2['tags'][:6]])
        
        prompt = f"""Compara brevemente a estos dos profesores del CUCEI basándote en las evaluaciones de estudiantes:

PROFESOR 1: {prof1}
Reseñas: {info1['num_comentarios']}
Tags frecuentes: {tags1_str}
Comentarios ejemplo:
{comentarios1}

PROFESOR 2: {prof2}
Reseñas: {info2['num_comentarios']}
Tags frecuentes: {tags2_str}
Comentarios ejemplo:
{comentarios2}

FORMATO REQUERIDO:
**{prof1}**
Estilo: [1 línea]
Fortalezas: [máximo 3 puntos]
Considerar: [máximo 2 puntos]

**{prof2}**
Estilo: [1 línea]
Fortalezas: [máximo 3 puntos]
Considerar: [máximo 2 puntos]

**Veredicto:**
Similitudes: [1 línea]
Diferencia clave: [1 línea]
Recomendación: [máximo 2 líneas]

Sé ultra conciso. Máximo 350 palabras."""

        return self.generar_respuesta_groq(prompt, max_tokens=450)
    
    def generar_respuesta(self, query: str, history) -> str:
        """Genera respuesta a una consulta del usuario"""
        if not query or not query.strip():
            return "Por favor escribe una pregunta."
        
        print(f"\n{'='*70}")
        print(f"Consulta: '{query}'")
        
        # Verificar cache
        cache_k = cache_key(query)
        if cache_k in RESPONSE_CACHE:
            print("Respuesta recuperada del cache")
            return RESPONSE_CACHE[cache_k]
        
        # Detectar comparacion
        if ' o ' in query.lower():
            print("Consulta de comparacion detectada")
            comparacion = self.buscar_comparacion(query)
            
            if comparacion and len(comparacion) >= 2:
                # Intentar generar comparación con Groq
                respuesta_llm = self.generar_comparacion_detallada(comparacion)
                
                if respuesta_llm:
                    print("Comparacion generada con Groq")
                    if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
                        RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
                    RESPONSE_CACHE[cache_k] = respuesta_llm
                    return respuesta_llm
                
                # Fallback: comparacion estructurada
                respuesta = "**Comparacion de Profesores**\n\n"
                for nombre, info in comparacion.items():
                    respuesta += f"**{nombre}**\n"
                    respuesta += f"{info['num_comentarios']} resenas analizadas\n\n"
                    
                    if info.get('tags') and len(info['tags']) > 0:
                        respuesta += f"**Características principales:**\n"
                        for tag, freq in info['tags'][:4]:
                            respuesta += f"  • {tag} ({freq} menciones)\n"
                    
                    if info.get('comentarios_muestra') and len(info['comentarios_muestra']) > 0:
                        respuesta += f"\n**Lo que dicen los estudiantes:**\n"
                        for i, com in enumerate(info['comentarios_muestra'][:2], 1):
                            comentario_corto = com[:120] + '...' if len(com) > 120 else com
                            respuesta += f"  {i}. \"{comentario_corto}\"\n"
                    
                    respuesta += "\n" + "─"*50 + "\n\n"
                
                respuesta += "Tip: Revisa las caracteristicas y comentarios para decidir cual se adapta mejor a tu estilo de aprendizaje."
                
                print("Comparacion estructurada generada")
                if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
                    RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
                RESPONSE_CACHE[cache_k] = respuesta.strip()
                return respuesta.strip()
            
            return "No se pudo comparar los profesores. Verifica que los nombres estén escritos correctamente."
        
        # Busqueda simple de profesor
        print("Busqueda simple de profesor")
        info = self.buscar_contexto(query)
        
        if not info:
            return "No se encontro informacion sobre ese profesor. Intenta ser mas especifico con el nombre o verifica la ortografia."
        
        print(f"Informacion encontrada para: {info['profesor']}")
        
        # Construir contexto
        contexto = f"Profesor: {info['profesor']}\n"
        
        if info.get('num_evaluaciones'):
            contexto += f"Numero de evaluaciones: {info['num_evaluaciones']}\n"
        
        if info.get('calificacion_promedio'):
            contexto += f"Calificacion Promedio: {info['calificacion_promedio']:.1f}/10\n"
        
        if info.get('tags'):
            contexto += f"Tags Comunes: {', '.join(info['tags'])}\n"
        
        if info.get('comentarios'):
            comentarios_str = '\n- '.join(info['comentarios'][:3])
            contexto += f"Comentarios Relevantes:\n- {comentarios_str}\n"
        
        # Generar respuesta con Groq
        prompt = f"""Contexto de evaluaciones de estudiantes:
---
{contexto}
---
Pregunta del estudiante: {query}

Responde de forma concisa (3-5 lineas maximo) basandote SOLO en el contexto proporcionado. 
Si la pregunta no puede responderse con el contexto, di "No tengo suficiente informacion sobre ese aspecto especifico"."""
        
        print("Generando respuesta con Groq...")
        response_text = self.generar_respuesta_groq(prompt, max_tokens=200)
        
        # Guardar en cache
        if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
            RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
        RESPONSE_CACHE[cache_k] = response_text
        
        print("Respuesta generada exitosamente")
        return response_text

def guardar_resena(profesor, materia, resena):
    """Guarda una nueva resena en CSV"""
    if not profesor or not profesor.strip():
        return "Error: El nombre del profesor no puede estar vacio."
    
    if not resena or not resena.strip():
        return "Error: La resena no puede estar vacia."
    
    archivo_csv = 'nuevas_resenas.csv'
    header = ['timestamp', 'profesor', 'materia', 'resena']
    file_exists = os.path.isfile(archivo_csv)
    
    try:
        with open(archivo_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, profesor.strip(), materia.strip(), resena.strip()])
        
        print(f"Nueva resena guardada para: {profesor.strip()}")
        return f"Exito: Tu resena para '{profesor.strip()}' ha sido guardada. Gracias por contribuir."
    
    except Exception as e:
        print(f"Error al guardar la resena: {e}")
        return f"Error: No se pudo guardar tu resena. Intenta nuevamente."

# Inicializar el bot
print("\n" + "="*70)
print("Inicializando sistema...")
print("="*70)

try:
    bot = ChatCUCEI()
    print("\nSistema inicializado correctamente")
    print("Bot listo para recibir consultas")
except Exception as e:
    print(f"\nERROR CRITICO: No se pudo inicializar el bot")
    print(f"Detalle: {e}")
    import traceback
    traceback.print_exc()
    bot = None

# Configurar tema de Gradio
theme = gr.themes.Base(
    primary_hue="neutral",
    secondary_hue="neutral",
    neutral_hue="slate"
).set(
    body_background_fill="#F8F9FA",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
    button_primary_background_fill="black",
    button_primary_text_color="white",
    button_secondary_background_fill="#F1F5F9",
    button_secondary_text_color="black",
)

# Crear interfaz Gradio
with gr.Blocks(theme=theme, title="Asesor Academico CUCEI") as demo:
    gr.Markdown("# Asesor Academico CUCEI")
    
    if not bot:
        gr.Markdown("""
        ## Error de Inicializacion
        El sistema no pudo iniciarse correctamente. Verifica:
        - Que exista el directorio `chroma_db/`
        - Que la variable `GROQ_API_KEY` este configurada
        - Los logs de error arriba para mas detalles
        """)
    
    with gr.Tabs():
        # Tab 1: Consultar
        with gr.TabItem("Consultar Asesor"):
            gr.Markdown("Haz una pregunta sobre los profesores del CUCEI.")
            
            chatbot_ui = gr.Chatbot(
                height=450, 
                label="Chat", 
                show_label=False
            )
            
            msg_input = gr.Textbox(
                label="Tu Pregunta", 
                placeholder="Ej: Que opinan del profesor Juan Perez para Ecuaciones Diferenciales?"
            )
            
            clear_button = gr.ClearButton([msg_input, chatbot_ui])
            
            def respond(message, chat_history):
                if not bot:
                    bot_message = "Error: El sistema no esta inicializado correctamente."
                    chat_history.append((message, bot_message))
                    return "", chat_history
                
                bot_message = bot.generar_respuesta(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history
            
            msg_input.submit(respond, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        
        # Tab 2: Anadir Resena
        with gr.TabItem("Anadir Resena"):
            gr.Markdown("Ayuda a otros estudiantes. Comparte tu experiencia sobre un profesor.")
            
            with gr.Column():
                profesor_input = gr.Textbox(
                    label="Nombre Completo del Profesor", 
                    placeholder="Ej: Dr. Juan Perez Gonzalez"
                )
                materia_input = gr.Textbox(
                    label="Materia (Opcional)", 
                    placeholder="Ej: Ecuaciones Diferenciales"
                )
                resena_input = gr.Textbox(
                    label="Tu Resena", 
                    placeholder="Describe tu experiencia, pros y contras, etc.", 
                    lines=5
                )
                submit_button = gr.Button("Enviar Resena", variant="primary")
                confirmation_output = gr.Textbox(label="Estado", interactive=False, lines=2)
            
            submit_button.click(
                fn=guardar_resena,
                inputs=[profesor_input, materia_input, resena_input],
                outputs=[confirmation_output]
            )

port = int(os.environ.get("PORT", 8080))

print(f"\nIniciando servidor en puerto {port}...")
print("Optimizado con Groq para respuestas rapidas")

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)