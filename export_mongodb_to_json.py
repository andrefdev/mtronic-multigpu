from MongoDBService import MongoDBService
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable global para el contador de progreso
progress_counter = 0
progress_lock = threading.Lock()


def process_uuid(uuid: str, mongo_service: MongoDBService) -> Optional[Dict]:
    """
    Procesa un UUID individual y retorna sus datos.

    Args:
        uuid: UUID del documento a procesar
        mongo_service: Instancia del servicio MongoDB

    Returns:
        Optional[Dict]: Datos del documento procesado o None si hay error
    """
    global progress_counter

    try:
        chunks = mongo_service.get_chunks_by_uuid(uuid)
        if chunks:
            document_data = {"uuid": uuid, "chunks": chunks}

            # Actualizar contador de progreso
            with progress_lock:
                global progress_counter
                progress_counter += 1
                logger.info(f"Procesado documento {progress_counter}: {uuid}")

            return document_data
    except Exception as e:
        logger.error(f"Error procesando UUID {uuid}: {str(e)}")
    return None


def export_mongodb_to_json(
    output_file: str = "mongodb_export.json", max_workers: int = 40
):
    """
    Exporta todos los documentos de MongoDB a un archivo JSON usando procesamiento paralelo.

    Args:
        output_file: Nombre del archivo de salida
        max_workers: Número máximo de hilos de trabajo
    """
    mongo_service = MongoDBService()

    try:
        # Obtener todos los UUIDs únicos
        uuids = mongo_service.get_unique_uuids()
        total_uuids = len(uuids)
        logger.info(f"Encontrados {total_uuids} documentos únicos para exportar")

        all_documents = []

        # Crear un pool de hilos para procesar UUIDs en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear un diccionario de futuros
            future_to_uuid = {
                executor.submit(process_uuid, uuid, mongo_service): uuid
                for uuid in uuids
            }

            # Procesar los resultados a medida que se completan
            for future in as_completed(future_to_uuid):
                uuid = future_to_uuid[future]
                try:
                    result = future.result()
                    if result:
                        all_documents.append(result)
                except Exception as e:
                    logger.error(f"Error procesando UUID {uuid}: {str(e)}")

        # Guardar en archivo JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Exportación completada. Archivo guardado como: {output_file}")
        logger.info(f"Total de documentos exportados: {len(all_documents)}")

    except Exception as e:
        logger.error(f"Error durante la exportación: {str(e)}")
    finally:
        mongo_service.close()


if __name__ == "__main__":
    # Ajustar el número de workers según tu CPU
    export_mongodb_to_json(max_workers=20)
