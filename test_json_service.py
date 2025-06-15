#!/usr/bin/env python3
"""
Script de prueba para verificar que el sistema funciona con archivos JSON
"""

import sys
import os
import logging

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from JSONDataService import JSONDataService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_json_service(json_file_path: str = "mongodb_export.json"):
    """Prueba el servicio JSON"""
    
    print(f"🔍 Probando JSONDataService con archivo: {json_file_path}")
    
    try:
        # Verificar si el archivo existe
        if not os.path.exists(json_file_path):
            print(f"❌ Archivo JSON no encontrado: {json_file_path}")
            print("💡 Ejecuta primero: python export_mongodb_to_json.py")
            return False
        
        # Crear servicio
        json_service = JSONDataService(json_file_path)
        
        # Obtener estadísticas
        stats = json_service.get_statistics()
        print(f"\n📊 Estadísticas del dataset:")
        print(f"   - Total de documentos: {stats['total_documents']:,}")
        print(f"   - Total de chunks: {stats['total_chunks']:,}")
        print(f"   - UUIDs únicos: {stats['unique_uuids']:,}")
        print(f"   - Promedio chunks por doc: {stats['average_chunks_per_doc']:.2f}")
        
        # Obtener UUIDs
        print(f"\n🔍 Obteniendo UUIDs...")
        uuids = json_service.get_uuids_sorted_by_insertion()
        print(f"✅ UUIDs obtenidos: {len(uuids)}")
        
        if uuids:
            print(f"\n📋 Primeros 3 UUIDs:")
            for i, uuid in enumerate(uuids[:3]):
                print(f"   {i+1}. {uuid}")
                
                # Obtener chunks del primer UUID
                if i == 0:
                    chunks = json_service.get_chunks_by_uuid(uuid)
                    print(f"      - Chunks: {len(chunks)}")
                    if chunks:
                        first_chunk = chunks[0]
                        content_preview = first_chunk.get('content', '')[:100]
                        print(f"      - Preview: {content_preview}...")
        
        # Limpiar
        json_service.close()
        
        print(f"\n✅ Prueba completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    
    # Buscar archivos JSON en el directorio
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if json_files:
        print(f"📁 Archivos JSON encontrados: {json_files}")
        
        # Usar el primero que contenga 'export' o 'mongodb'
        json_file = None
        for f in json_files:
            if 'export' in f.lower() or 'mongodb' in f.lower():
                json_file = f
                break
        
        if not json_file:
            json_file = json_files[0]
        
        print(f"🎯 Usando archivo: {json_file}")
        success = test_json_service(json_file)
    else:
        print("❌ No se encontraron archivos JSON")
        print("💡 Ejecuta primero: python export_mongodb_to_json.py")
        success = False
    
    if success:
        print("\n🚀 El sistema está listo para usar JSON en lugar de MongoDB")
        print("💡 Ahora puedes ejecutar: python bge_m3_multi_gpu_optimized.py")
    else:
        print("\n❌ Hay problemas que necesitan ser resueltos")

if __name__ == "__main__":
    main()
