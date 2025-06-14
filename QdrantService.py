from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http import models
import logging
from typing import Optional, Dict, List
import uuid
from dotenv import load_dotenv
import os

load_dotenv()


class QdrantService:
    def __init__(
        self,
        host: str = "sv1.mecatronicmaquinariaperu.com",
        port: int = 6333,
        api_key: str = str(os.getenv("QDRANT_API_KEY")),
    ):
        """
        Inicializa el servicio de Qdrant.

        Args:
            host: Host de Qdrant
            port: Puerto de Qdrant
            api_key: API key de Qdrant
        """
        self.logger = logging.getLogger(__name__)
        self.client = QdrantClient(
            host=host, port=port, api_key=api_key, https=False, timeout=60
        )
        self.collection_name = "mtronic_knowledge_base_dot"

        # Asegurar que la colección existe con dimensionalidad 768
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE, on_disk=True
                ),
            )

    def check_pdf_exists(self, uuid: str) -> bool:
        """
        Verifica si un PDF ya existe en Qdrant basado en el UUID del archivo.

        Args:
            uuid: UUID del archivo PDF

        Returns:
            bool: True si el PDF existe, False en caso contrario
        """
        try:
            # Buscar puntos que coincidan con el UUID del archivo en el payload
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="uuid", match=models.MatchValue(value=uuid)
                        )
                    ]
                ),
                limit=1,
                with_vectors=False,
                with_payload=True,
            )

            return bool(response[0])

        except Exception as e:
            self.logger.error(f"Error verificando PDF en Qdrant: {str(e)}")
            return False

    def store_embedding(
        self, txt_content: str, embedding: List[float], metadata: Optional[Dict] = None
    ) -> bool:
        """
        Almacena un embedding en Qdrant.

        Args:
            txt_content: Contenido de texto del PDF
            embedding: Vector de embedding
            metadata: Metadatos asociados (opcional)

        Returns:
            bool: True si se almacenó correctamente, False en caso contrario
        """
        if metadata is None:
            metadata = {}

        try:
            point_id = str(uuid.uuid4())

            # Crear el punto con el embedding y payload
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={**metadata, "content": txt_content},
            )

            # Insertar el punto en Qdrant
            self.client.upsert(collection_name=self.collection_name, points=[point])

            return True

        except Exception as e:
            self.logger.error(f"Error al almacenar embedding en Qdrant: {str(e)}")
            return False
