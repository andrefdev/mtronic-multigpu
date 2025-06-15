# BGE-M3 Multi-GPU con JSON Data Source

## Cambios Realizados

El sistema ha sido modificado para usar archivos JSON en lugar de conectarse directamente a MongoDB, lo que mejora el rendimiento y simplifica el despliegue.

## Arquitectura Actualizada

```
MongoDB → export_mongodb_to_json.py → JSON File → bge_m3_multi_gpu_optimized.py → Qdrant
```

## Archivos Modificados

- **`JSONDataService.py`** - Nueva clase para manejar datos desde JSON
- **`bge_m3_multi_gpu_optimized.py`** - Actualizado para usar JSON en lugar de MongoDB
- **`requirements_multi_gpu.txt`** - Eliminadas dependencias de MongoDB
- **`test_json_service.py`** - Script de prueba para el nuevo sistema

## Instrucciones de Uso

### Paso 1: Exportar datos de MongoDB a JSON

```bash
# Ejecutar el script de exportación
python export_mongodb_to_json.py
```

Esto creará un archivo `mongodb_export.json` con todos los datos.

### Paso 2: Verificar el archivo JSON

```bash
# Probar que el archivo JSON funciona correctamente
python test_json_service.py
```

### Paso 3: Procesar embeddings con BGE-M3

```bash
# Ejecutar el script principal (ahora usa JSON)
python bge_m3_multi_gpu_optimized.py
```

## Configuración

El archivo JSON se configura en la clase `ProcessingConfig`:

```python
@dataclass
class ProcessingConfig:
    # ... otras configuraciones ...
    json_file_path: str = "mongodb_export.json"  # Ruta al archivo JSON
```

## Ventajas del Nuevo Sistema

### ✅ **Beneficios:**
- **Rendimiento**: Sin latencia de red con MongoDB
- **Simplicidad**: No necesita mantener conexión con MongoDB
- **Portabilidad**: Fácil de mover entre servidores
- **Debugging**: Fácil inspección de datos
- **Atomicidad**: Los datos no cambian durante el procesamiento

### 📊 **Comparación de Rendimiento:**

| Métrica | MongoDB | JSON |
|---------|---------|------|
| Tiempo de inicio | ~5-10s | ~1-2s |
| Latencia por query | ~10-50ms | ~0.1ms |
| Dependencias | pymongo, motor | Ninguna |
| Conexiones | Persistent | Ninguna |

## Formato del Archivo JSON

```json
[
  {
    "uuid": "document-uuid-1",
    "chunks": [
      {
        "content": "Texto del chunk...",
        "metadata": {
          "uuid": "document-uuid-1",
          "chunk_index": 0,
          "total_chunks": 5,
          "filename": "documento.pdf",
          "page": 1
        }
      }
    ]
  }
]
```

## Troubleshooting

### Archivo JSON no encontrado
```bash
❌ Archivo JSON no encontrado: mongodb_export.json
💡 Ejecuta primero: python export_mongodb_to_json.py
```

### Archivo JSON corrupto
```bash
# Verificar estructura del JSON
python -c "
import json
with open('mongodb_export.json', 'r') as f:
    data = json.load(f)
    print(f'Documentos: {len(data)}')
    print(f'Primer UUID: {data[0][\"uuid\"] if data else \"None\"}')
"
```

### Problema de memoria con archivos grandes
Si el archivo JSON es muy grande (>8GB), considera:

1. **Dividir en chunks más pequeños**
2. **Usar streaming JSON parsing**
3. **Procesamiento por lotes**

## Scripts de Utilidad

- **`export_mongodb_to_json.py`** - Exporta MongoDB a JSON
- **`test_json_service.py`** - Prueba el servicio JSON
- **`diagnose_mongodb.py`** - Diagnóstico de MongoDB (legacy)
- **`test_mongo_simple.py`** - Prueba simple de MongoDB (legacy)

## Monitoreo

El sistema incluye logging detallado:

```
📁 Cargando datos desde: mongodb_export.json
✅ Datos cargados exitosamente: 1,234 documentos
📊 UUIDs únicos: 1,234
🚀 Procesando 1,234 documentos con 4x L40S GPUs
```

## Migración desde MongoDB

Si tienes datos en MongoDB y quieres migrar:

1. Ejecuta `export_mongodb_to_json.py`
2. Verifica con `test_json_service.py`
3. Actualiza `ProcessingConfig.json_file_path` si es necesario
4. Ejecuta el script principal

El sistema es completamente compatible con los datos existentes.
