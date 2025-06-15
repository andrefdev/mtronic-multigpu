#!/usr/bin/env python3
"""
Script simple para probar MongoDB y identificar el problema
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MongoDBService import MongoDBService
import logging

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("🔍 Iniciando prueba de MongoDB...")
    
    try:
        # Crear servicio
        mongo = MongoDBService()
        
        # Prueba 1: Contar documentos
        print("\n📊 Contando documentos...")
        total = mongo.collection.count_documents({})
        print(f"Total de documentos: {total}")
        
        if total == 0:
            print("❌ No hay documentos en la colección")
            return
        
        # Prueba 2: Obtener un documento de muestra
        print("\n📄 Obteniendo documento de muestra...")
        sample = mongo.collection.find_one({})
        if sample:
            print("✅ Documento encontrado")
            print(f"Claves: {list(sample.keys())}")
            if 'metadata' in sample:
                print(f"Metadata claves: {list(sample['metadata'].keys())}")
                if 'uuid' in sample['metadata']:
                    print(f"UUID ejemplo: {sample['metadata']['uuid']}")
                else:
                    print("❌ No hay campo 'uuid' en metadata")
            else:
                print("❌ No hay campo 'metadata'")
        
        # Prueba 3: Contar con metadata.uuid
        print("\n🔍 Contando documentos con metadata.uuid...")
        with_uuid = mongo.collection.count_documents({"metadata.uuid": {"$exists": True}})
        print(f"Documentos con metadata.uuid: {with_uuid}")
        
        # Prueba 4: Obtener UUIDs
        print("\n📝 Obteniendo UUIDs...")
        uuids = mongo.get_uuids_sorted_by_insertion()
        print(f"UUIDs encontrados: {len(uuids)}")
        
        if uuids:
            print("✅ UUIDs obtenidos correctamente")
            print(f"Primeros 3: {uuids[:3]}")
        else:
            print("❌ No se obtuvieron UUIDs")
        
        # Cerrar conexión
        mongo.close()
        print("\n✅ Prueba completada")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
