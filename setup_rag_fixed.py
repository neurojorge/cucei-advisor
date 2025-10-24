import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
import time
import shutil

class IndexadorChatCUCEI:
    def __init__(self, persist_dir="./chroma_db", force_recreate=False):
        """
        Inicializa el sistema de indexación
        
        Args:
            persist_dir: Directorio donde se guardará ChromaDB
            force_recreate: Si True, elimina y recrea la BD
        """
        print("="*70)
        print("INDEXADOR ChatCUCEI - Sistema RAG Optimizado (FIXED)")
        print("="*70)
        
        # Si force_recreate, eliminar BD corrupta
        if force_recreate:
            persist_path = Path(persist_dir)
            if persist_path.exists():
                print(f"\n⚠️  ELIMINANDO BD CORRUPTA: {persist_dir}")
                shutil.rmtree(persist_path)
                print("✅ BD eliminada")
        
        # Cargar modelo de embeddings
        print("\n📦 Cargando modelo de embeddings...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Modelo cargado: paraphrase-multilingual-MiniLM-L12-v2")
        
        # Inicializar ChromaDB con configuración robusta
        print(f"\n🔧 Inicializando ChromaDB en: {persist_dir}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Intentar obtener o crear colección
            try:
                self.collection = self.client.get_collection("profesores_cucei")
                count = self.collection.count()
                print(f"✅ Colección existente: {count} documentos")
                
                if count == 0 and not force_recreate:
                    print("⚠️  Colección vacía detectada")
                    
            except Exception:
                print("📝 Creando nueva colección...")
                self.collection = self.client.create_collection(
                    name="profesores_cucei",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ Nueva colección creada")
                
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO con ChromaDB: {e}")
            print("\n💡 SOLUCIÓN: Elimina manualmente la carpeta './chroma_db' y ejecuta de nuevo")
            raise
        
        print()
    
    @staticmethod
    def normalizar_nombre(nombre: str) -> str:
        """Normaliza nombre del profesor"""
        # Eliminar comillas
        nombre = nombre.replace('"', '').replace("'", '')
        
        # Normalizar espacios y comas
        nombre = nombre.replace(',', ' ')
        nombre = ' '.join(nombre.split())
        
        # Mayúsculas
        nombre = nombre.upper()
        
        # Remover acentos
        nombre = unicodedata.normalize('NFKD', nombre)
        nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
        
        return nombre.strip()
    
    def limpiar_comentario(self, comentario: str) -> Optional[str]:
        """Limpia y valida comentarios"""
        if pd.isna(comentario) or not comentario:
            return None
        
        comentario = str(comentario).strip()
        
        # Remover calificación del inicio
        comentario = re.sub(r'^Calificación:\s*\d+\.?\d*/10\s*-?\s*', '', comentario)
        
        # Filtrar comentarios inválidos
        if comentario in ['(No hay información del maestro)', '', 'N/A', 'nan']:
            return None
        
        if len(comentario) < 10:
            return None
        
        return comentario.strip()
    
    def extraer_calificacion(self, texto: str) -> float:
        """Extrae calificación del texto"""
        if not isinstance(texto, str):
            return 0.0
        match = re.search(r'Calificación:\s*(\d+\.?\d*)/10', texto)
        return float(match.group(1)) if match else 0.0
    
    def extraer_tags(self, texto: str, max_tags: int = 5) -> List[str]:
        """Extrae tags del comentario"""
        if not isinstance(texto, str) or " - " not in texto:
            return []
        
        # Extraer parte de tags después del guión
        tags_text = texto.split(" - ", 1)[1]
        
        # Separar y limpiar tags
        tags = []
        for tag in tags_text.split(","):
            tag = tag.strip().upper()
            if tag and len(tag) > 2:
                tags.append(tag)
        
        return tags[:max_tags]
    
    def crear_documento_embedding(self, row: pd.Series, comentario_limpio: str) -> str:
        """Crea documento para embedding"""
        profesor = row['PROFESOR']
        materia = row['MATERIA'] if pd.notna(row['MATERIA']) else ""
        depto = row['DEPARTAMENTO'] if pd.notna(row['DEPARTAMENTO']) else ""
        calificacion = self.extraer_calificacion(row['COMENTARIOS'])
        tags = self.extraer_tags(row['COMENTARIOS'])
        
        # Construir documento rico para búsqueda
        documento = f"""
        Profesor: {profesor}
        Nombre del maestro: {profesor}
        Evaluación del profesor {profesor}
        Materia: {materia}
        Departamento: {depto}
        Calificación: {calificacion}/10
        Características: {', '.join(tags) if tags else 'N/A'}
        Comentarios: {comentario_limpio}
        """.strip()
        
        return documento
    
    def indexar_csv(self, csv_path: str, batch_size: int = 100):
        """Indexa datos del CSV en ChromaDB"""
        print(f"{'='*70}")
        print("🚀 INICIANDO INDEXACIÓN")
        print(f"{'='*70}\n")
        
        # Validar archivo
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"❌ No se encuentra: {csv_path}")
        
        print(f"📄 Archivo: {csv_path.name}")
        
        # Leer CSV
        print("📖 Leyendo CSV...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"   Total filas: {len(df):,}")
        
        # Preparar datos
        documentos = []
        metadatas = []
        ids = []
        
        stats = {
            'total': len(df),
            'procesados': 0,
            'omitidos': 0,
            'sin_comentarios': 0,
            'sin_profesor': 0
        }
        
        print("\n⚙️  Procesando datos...")
        inicio = time.time()
        
        for idx, row in df.iterrows():
            # Validar profesor
            if pd.isna(row['PROFESOR']) or not str(row['PROFESOR']).strip():
                stats['sin_profesor'] += 1
                stats['omitidos'] += 1
                continue
            
            profesor = str(row['PROFESOR']).strip()
            profesor_normalizado = self.normalizar_nombre(profesor)
            
            # Validar comentario
            comentario_limpio = self.limpiar_comentario(row['COMENTARIOS'])
            if not comentario_limpio:
                stats['sin_comentarios'] += 1
                stats['omitidos'] += 1
                continue
            
            # Extraer metadata
            comentarios_original = str(row['COMENTARIOS'])
            calificacion = self.extraer_calificacion(comentarios_original)
            tags = self.extraer_tags(comentarios_original)
            
            materia = str(row['MATERIA']) if pd.notna(row['MATERIA']) else ""
            depto = str(row['DEPARTAMENTO']) if pd.notna(row['DEPARTAMENTO']) else ""
            division = str(row['DIVISION']) if 'DIVISION' in row and pd.notna(row['DIVISION']) else ""
            
            # Crear documento
            documento = self.crear_documento_embedding(row, comentario_limpio)
            documentos.append(documento)
            
            # CRÍTICO: Asegurar que calificacion sea float, no int
            metadatas.append({
                "profesor": profesor,
                "profesor_normalizado": profesor_normalizado,
                "materia": materia,
                "departamento": depto,
                "division": division,
                "calificacion": float(calificacion),  # CONVERTIR A FLOAT
                "tags": ", ".join(tags) if tags else "",
                "comentarios": comentario_limpio,
                "comentarios_original": comentarios_original
            })
            
            ids.append(f"resena_{idx}")
            stats['procesados'] += 1
            
            # Log progreso
            if (idx + 1) % 500 == 0:
                print(f"   Procesados: {idx + 1:,}/{len(df):,}")
        
        print(f"\n✅ Procesamiento completado en {time.time() - inicio:.2f}s")
        print(f"\n📊 Resumen:")
        print(f"   Total filas: {stats['total']:,}")
        print(f"   ✅ Procesados: {stats['procesados']:,}")
        print(f"   ⚠️  Omitidos: {stats['omitidos']:,}")
        print(f"     - Sin profesor: {stats['sin_profesor']:,}")
        print(f"     - Sin comentarios válidos: {stats['sin_comentarios']:,}")
        
        # Indexar en ChromaDB
        print(f"\n🔍 Generando embeddings e indexando en ChromaDB...")
        print(f"   Batch size: {batch_size}")
        
        inicio = time.time()
        
        for i in range(0, len(documentos), batch_size):
            batch_docs = documentos[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(documentos):
                current = min(i + batch_size, len(documentos))
                print(f"   ✓ {current:,}/{len(documentos):,} indexados")
        
        tiempo_indexacion = time.time() - inicio
        
        print(f"\n✅ Indexación completada en {tiempo_indexacion:.2f}s")
        print(f"📚 Total documentos en ChromaDB: {self.collection.count():,}")
        print(f"⚡ Velocidad: {len(documentos)/tiempo_indexacion:.1f} docs/segundo")
    
    def verificar_indexacion(self, n_samples: int = 5):
        """Verifica que la indexación fue exitosa"""
        print(f"\n{'='*70}")
        print("🔍 VERIFICACIÓN DE INDEXACIÓN")
        print(f"{'='*70}\n")
        
        total = self.collection.count()
        print(f"📚 Total documentos: {total:,}\n")
        
        if total == 0:
            print("⚠️  Base de datos vacía")
            return
        
        # Obtener muestra
        peek = self.collection.peek(limit=n_samples)
        
        print(f"📋 Muestra de {n_samples} documentos:\n")
        
        for i, (doc, meta) in enumerate(zip(peek['documents'], peek['metadatas']), 1):
            print(f"{'─'*70}")
            print(f"Documento #{i}")
            print(f"{'─'*70}")
            print(f"👤 Profesor: {meta['profesor']}")
            print(f"🔤 Normalizado: {meta['profesor_normalizado']}")
            print(f"⭐ Calificación: {meta['calificacion']}/10")
            if meta['tags']:
                tags_preview = ', '.join(meta['tags'].split(',')[:3])
                print(f"🏷️  Tags: {tags_preview}")
            if meta['materia']:
                print(f"📖 Materia: {meta['materia']}")
            print(f"💬 Comentario: {meta['comentarios'][:100]}...")
            print()
        
        print(f"{'─'*70}")
        print("📊 Estadísticas generales:")
        print(f"   Total profesores únicos: ~{total//5} (estimado)")
        print(f"   Promedio reseñas/profesor: ~5")
        print(f"{'─'*70}\n")


def main():
    """Script principal"""
    CSV_PATH = r"C:\Users\morde\Desktop\CUCEI-ADVISOR\evaluaciones_con_departamentos.csv"
    CHROMA_DB_PATH = "./chroma_db"
    
    print("\n" + "="*70)
    print("⚙️  CONFIGURACIÓN")
    print("="*70)
    print(f"CSV: {CSV_PATH}")
    print(f"ChromaDB: {CHROMA_DB_PATH}")
    print()
    
    # IMPORTANTE: force_recreate=True para regenerar BD corrupta
    indexador = IndexadorChatCUCEI(
        persist_dir=CHROMA_DB_PATH,
        force_recreate=True  # ← ESTO ELIMINA LA BD CORRUPTA
    )
    
    # Indexar
    indexador.indexar_csv(CSV_PATH)
    
    # Verificar
    indexador.verificar_indexacion(n_samples=10)
    
    # Mensaje final
    print(f"\n{'='*70}")
    print("✅ INDEXACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"\n📂 Base de datos lista en: {CHROMA_DB_PATH}")
    print(f"📚 Total documentos: {indexador.collection.count():,}")
    print(f"\n🚀 SIGUIENTE PASO:")
    print(f"   Ejecuta tu app.py:")
    print(f"   python app.py\n")


if __name__ == "__main__":
    main()