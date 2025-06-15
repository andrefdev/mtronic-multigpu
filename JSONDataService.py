import json
import logging
from typing import Dict, List, Optional
import os


class JSONDataService:
    """Servicio para leer datos desde un archivo JSON exportado de MongoDB"""
    
    def __init__(self, json_file_path: str = "mongodb_export.json"):
        self.logger = logging.getLogger(__name__)
        self.json_file_path = json_file_path
        self.data = []
        self.uuid_to_chunks = {}
        self.load_data()
    
    def load_data(self):
        """Carga los datos desde el archivo JSON"""
        try:
            if not os.path.exists(self.json_file_path):
                self.logger.error(f"❌ Archivo JSON no encontrado: {self.json_file_path}")
                raise FileNotFoundError(f"Archivo {self.json_file_path} no existe")
            
            self.logger.info(f"📁 Cargando datos desde: {self.json_file_path}")
            
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Crear índice UUID -> chunks para acceso rápido
            for document in self.data:
                uuid = document.get('uuid')
                chunks = document.get('chunks', [])
                if uuid:
                    self.uuid_to_chunks[uuid] = chunks
            
            self.logger.info(f"✅ Datos cargados exitosamente: {len(self.data)} documentos")
            self.logger.info(f"📊 UUIDs únicos: {len(self.uuid_to_chunks)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando archivo JSON: {e}")
            raise
    
    def get_uuids_sorted_by_insertion(self) -> List[str]:
        """
        Obtiene UUIDs únicos en el orden que aparecen en el JSON
        (asumiendo que ya están ordenados por inserción)
        
        Returns:
            List[str]: Lista de UUIDs ordenados
        """
        try:
            uuids = list(self.uuid_to_chunks.keys())
            self.logger.info(f"📝 UUIDs obtenidos desde JSON: {len(uuids)}")
            return uuids
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo UUIDs: {e}")
            return []
    
    def get_unique_uuids(self) -> List[str]:
        """
        Obtiene todos los UUIDs únicos de los documentos.
        
        Returns:
            List[str]: Lista de UUIDs únicos
        """
        return self.get_uuids_sorted_by_insertion()
    
    def get_chunks_by_uuid(self, uuid: str) -> List[Dict]:
        """
        Obtiene todos los chunks de un UUID específico.
        
        Args:
            uuid: UUID del archivo
        
        Returns:
            List[Dict]: Lista de chunks con su contenido y metadata
        """
        try:
            chunks = self.uuid_to_chunks.get(uuid, [])
            if not chunks:
                self.logger.warning(f"⚠️  No se encontraron chunks para UUID: {uuid}")
            return chunks
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo chunks para UUID {uuid}: {e}")
            return []
    
    def check_pdf_exists(self, uuid: str) -> bool:
        """
        Verifica si un PDF existe basado en el UUID.
        
        Args:
            uuid: UUID del archivo PDF
        
        Returns:
            bool: True si el PDF existe, False en caso contrario
        """
        return uuid in self.uuid_to_chunks
    
    def get_document_count(self) -> int:
        """
        Obtiene el número total de documentos.
        
        Returns:
            int: Número total de documentos
        """
        return len(self.data)
    
    def get_total_chunks_count(self) -> int:
        """
        Obtiene el número total de chunks en todos los documentos.
        
        Returns:
            int: Número total de chunks
        """
        total = sum(len(chunks) for chunks in self.uuid_to_chunks.values())
        return total
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas del dataset.
        
        Returns:
            Dict: Estadísticas del dataset
        """
        stats = {
            "total_documents": self.get_document_count(),
            "total_chunks": self.get_total_chunks_count(),
            "unique_uuids": len(self.uuid_to_chunks),
            "average_chunks_per_doc": 0
        }
        
        if stats["total_documents"] > 0:
            stats["average_chunks_per_doc"] = stats["total_chunks"] / stats["total_documents"]
        
        return stats
    
    def close(self):
        """Limpia los datos cargados (equivalente al close de MongoDB)"""
        self.data = []
        self.uuid_to_chunks = {}
        self.logger.info("🔒 Datos JSON limpiados de memoria")
