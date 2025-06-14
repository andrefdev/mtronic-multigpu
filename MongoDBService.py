from pymongo import MongoClient
from typing import Dict, List, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()


class MongoDBService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mongo_uri = os.getenv(
            "MONGODB_URI",
            "mongodb+srv://admin:HaOP1lBpteLcT0In@cluster1.upwms3u.mongodb.net/",
        )
        self.db_name = os.getenv("MONGODB_DB", "file_contents")
        self.collection_name = "pdf_chunks"

        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.logger.info("Conexión exitosa a MongoDB")
        except Exception as e:
            self.logger.error(f"Error conectando a MongoDB: {str(e)}")
            raise

    def check_pdf_exists(self, uuid: str) -> bool:
        """
        Verifica si un PDF ya existe en MongoDB basado en el UUID del archivo.

        Args:
            uuid: UUID del archivo PDF

        Returns:
            bool: True si el PDF existe, False en caso contrario
        """
        try:
            count = self.collection.count_documents({"metadata.uuid": uuid})
            return count > 0
        except Exception as e:
            self.logger.error(f"Error verificando PDF en MongoDB: {str(e)}")
            return False

    def store_chunk(
        self, chunk_content: str, metadata: Dict, chunk_index: int, total_chunks: int
    ) -> bool:
        """
        Almacena un chunk de texto y sus metadatos en MongoDB.

        Args:
            chunk_content: Contenido del chunk de texto
            metadata: Metadatos asociados al PDF
            chunk_index: Índice del chunk actual
            total_chunks: Total de chunks en el PDF

        Returns:
            bool: True si se almacenó correctamente, False en caso contrario
        """
        try:
            document = {
                "content": chunk_content,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                },
            }

            result = self.collection.insert_one(document)
            return bool(result.inserted_id)
        except Exception as e:
            self.logger.error(f"Error al almacenar chunk en MongoDB: {str(e)}")
            return False

    def get_uuids_sorted_by_insertion(self) -> List[str]:
        """
        Obtiene UUIDs únicos ordenados por orden de inserción descendente (más recientes primero)

        Returns:
            List[str]: Lista de UUIDs ordenados
        """
        try:
            # Usamos aggregate para obtener UUIDs únicos ordenados por $natural descendente
            pipeline = [
                {"$sort": {"$natural": -1}},
                {"$group": {"_id": "$metadata.uuid"}},
                {"$project": {"_id": 0, "uuid": "$_id"}},
            ]

            results = list(self.collection.aggregate(pipeline))
            return [doc["uuid"] for doc in results]
        except Exception as e:
            self.logger.error(f"Error obteniendo UUIDs ordenados: {str(e)}")
            return []

    def get_unique_uuids(self) -> List[str]:
        """
        Obtiene todos los UUIDs únicos de los documentos en la colección.

        Returns:
            List[str]: Lista de UUIDs únicos
        """
        try:
            pipeline = [
                {"$group": {"_id": "$metadata.uuid"}},
                {"$project": {"_id": 0, "uuid": "$_id"}},
            ]
            results = list(self.collection.aggregate(pipeline))
            return [doc["uuid"] for doc in results]
        except Exception as e:
            self.logger.error(f"Error obteniendo UUIDs únicos: {str(e)}")
            return []

    def get_chunks_by_uuid(self, uuid: str) -> List[Dict]:
        """
        Obtiene todos los chunks de un PDF específico.

        Args:
            uuid: UUID del archivo PDF

        Returns:
            List[Dict]: Lista de chunks con su contenido y metadata
        """
        try:
            chunks = list(
                self.collection.find(
                    {"metadata.uuid": uuid}, {"_id": 0, "content": 1, "metadata": 1}
                ).sort("metadata.chunk_index", 1)
            )
            return chunks
        except Exception as e:
            self.logger.error(f"Error obteniendo chunks de MongoDB: {str(e)}")
            return []

    def close(self):
        """Cierra la conexión con MongoDB"""
        try:
            self.client.close()
            self.logger.info("Conexión a MongoDB cerrada")
        except Exception as e:
            self.logger.error(f"Error cerrando conexión a MongoDB: {str(e)}")
