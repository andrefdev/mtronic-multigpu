#!/usr/bin/env python3
"""
Script de diagnóstico para verificar la conexión y contenido de MongoDB
"""

import logging
from MongoDBService import MongoDBService
from dotenv import load_dotenv
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal de diagnóstico"""
    logger.info("🔍 Iniciando diagnóstico de MongoDB...")
    
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Mostrar configuración
        mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://admin:HaOP1lBpteLcT0In@cluster1.upwms3u.mongodb.net/")
        db_name = os.getenv("MONGODB_DB", "file_contents")
        
        logger.info(f"📋 Configuración:")
        logger.info(f"   - URI: {mongo_uri[:50]}...")
        logger.info(f"   - DB: {db_name}")
        logger.info(f"   - Collection: pdf_chunks")
        
        # Crear servicio de MongoDB
        mongo_service = MongoDBService()
        
        # Verificar conexión
        logger.info("🔗 Verificando conexión...")
        
        # Contar documentos totales
        total_docs = mongo_service.collection.count_documents({})
        logger.info(f"📊 Total de documentos en la colección: {total_docs}")
        
        if total_docs == 0:
            logger.warning("⚠️  No hay documentos en la colección 'pdf_chunks'")
            return
        
        # Obtener estadísticas de la colección
        logger.info("📈 Estadísticas de la colección:")
        stats = mongo_service.db.command("collStats", "pdf_chunks")
        logger.info(f"   - Tamaño: {stats.get('size', 0):,} bytes")
        logger.info(f"   - Documentos: {stats.get('count', 0):,}")
        logger.info(f"   - Índices: {stats.get('nindexes', 0)}")
        
        # Obtener UUIDs únicos
        logger.info("🔍 Obteniendo UUIDs únicos...")
        uuids = mongo_service.get_unique_uuids()
        logger.info(f"📝 UUIDs únicos encontrados: {len(uuids)}")
        
        if uuids:
            logger.info("📋 Primeros 5 UUIDs:")
            for i, uuid in enumerate(uuids[:5]):
                logger.info(f"   {i+1}. {uuid}")
        
        # Obtener UUIDs ordenados por inserción
        logger.info("🔍 Obteniendo UUIDs ordenados por inserción...")
        sorted_uuids = mongo_service.get_uuids_sorted_by_insertion()
        logger.info(f"📝 UUIDs ordenados encontrados: {len(sorted_uuids)}")
        
        if sorted_uuids:
            logger.info("📋 Primeros 5 UUIDs ordenados:")
            for i, uuid in enumerate(sorted_uuids[:5]):
                logger.info(f"   {i+1}. {uuid}")
        
        # Verificar estructura de documentos
        logger.info("🔍 Verificando estructura de documentos...")
        sample_doc = mongo_service.collection.find_one({})
        
        if sample_doc:
            logger.info("📄 Estructura del primer documento:")
            logger.info(f"   - Claves principales: {list(sample_doc.keys())}")
            
            if 'metadata' in sample_doc:
                metadata = sample_doc['metadata']
                logger.info(f"   - Claves de metadata: {list(metadata.keys())}")
                
                if 'uuid' in metadata:
                    logger.info(f"   - UUID ejemplo: {metadata['uuid']}")
                else:
                    logger.warning("   ⚠️  No se encontró campo 'uuid' en metadata")
            else:
                logger.warning("   ⚠️  No se encontró campo 'metadata'")
        
        # Probar queries específicas
        logger.info("🔍 Probando queries específicas...")
        
        # Query por metadata.uuid
        test_query = {"metadata.uuid": {"$exists": True}}
        docs_with_uuid = mongo_service.collection.count_documents(test_query)
        logger.info(f"📊 Documentos con metadata.uuid: {docs_with_uuid}")
        
        # Query por metadata existente
        test_query2 = {"metadata": {"$exists": True}}
        docs_with_metadata = mongo_service.collection.count_documents(test_query2)
        logger.info(f"📊 Documentos con metadata: {docs_with_metadata}")
        
        logger.info("✅ Diagnóstico completado exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error durante el diagnóstico: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        try:
            mongo_service.close()
        except:
            pass

if __name__ == "__main__":
    main()
