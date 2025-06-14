# BGE-M3 Multi-GPU Optimized Processing

Sistema optimizado para generar embeddings BGE-M3 utilizando **4x L40S GPUs** con máximo rendimiento y carga concurrente a Qdrant.

## 🚀 Características Principales

- **Multi-GPU Processing**: Utiliza las 4x L40S GPUs simultáneamente
- **Procesamiento Masivo**: Batches de 512 por GPU (2048 total)
- **Upload Asíncrono**: 16 workers para subida concurrente a Qdrant
- **Pipeline Optimizado**: 3 etapas de pipeline para máximo throughput
- **Monitoreo en Tiempo Real**: Monitor visual de GPUs y rendimiento
- **Gestión Inteligente de Memoria**: Auto-limpieza y optimización VRAM
- **Configuración Adaptativa**: Ajustado específicamente para L40S (48GB VRAM)

## 📊 Rendimiento Esperado

Con la configuración optimizada para 4x L40S:

- **Throughput**: 500-1000+ chunks/segundo
- **Batch Size Total**: 2048 embeddings simultáneos
- **VRAM Utilizada**: ~170GB de 192GB disponibles
- **Procesamiento**: 10-50 documentos/minuto (según tamaño)

## 🔧 Instalación y Configuración

### 1. Configuración Automática

```bash
# Ejecutar script de configuración
python setup_multi_gpu.py
```

Este script:
- ✅ Verifica hardware (4x L40S)
- ✅ Instala dependencias optimizadas
- ✅ Configura PyTorch con CUDA 12.1
- ✅ Crea archivos de configuración
- ✅ Prepara variables de entorno

### 2. Configuración Manual (Alternativa)

```bash
# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install -r requirements_multi_gpu.txt

# Cargar variables de entorno
source .env.multi_gpu  # Linux/Mac
# o importar manualmente en Windows
```

## 🚀 Ejecución

### Método 1: Launcher Optimizado (Recomendado)

```bash
python launch_multi_gpu.py
```

### Método 2: Ejecución Directa

```bash
python bge_m3_multi_gpu_optimized.py
```

### Método 3: Con Monitor en Tiempo Real

```bash
# Terminal 1: Ejecutar procesamiento
python launch_multi_gpu.py

# Terminal 2: Monitor en tiempo real
python gpu_monitor.py
```

## 📁 Archivos del Sistema

```
├── bge_m3_multi_gpu_optimized.py    # Script principal optimizado
├── requirements_multi_gpu.txt        # Dependencias
├── setup_multi_gpu.py              # Configuración automática
├── launch_multi_gpu.py              # Launcher optimizado
├── gpu_monitor.py                   # Monitor en tiempo real
├── multi_gpu_config.json            # Configuración del sistema
├── .env.multi_gpu                   # Variables de entorno
└── bge_m3_multi_gpu.log            # Logs de procesamiento
```

## ⚙️ Configuración Avanzada

### Ajuste de Rendimiento

Editar `multi_gpu_config.json`:

```json
{
  "processing": {
    "batch_size_per_gpu": 512,      // Ajustar según VRAM disponible
    "num_workers_per_gpu": 4,       // Workers por GPU
    "async_upload_workers": 16,     // Workers de upload a Qdrant
    "pipeline_depth": 3,            // Profundidad del pipeline
    "memory_threshold": 0.85        // Límite de memoria (85%)
  }
}
```

### Variables de Entorno Críticas

```bash
# 4 GPUs disponibles
CUDA_VISIBLE_DEVICES=0,1,2,3

# Optimizaciones CUDA
CUDA_LAUNCH_BLOCKING=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CPU threads (32 cores / 4 GPUs)
OMP_NUM_THREADS=8
```

## 📊 Monitoreo y Diagnósticos

### Monitor en Tiempo Real

```bash
python gpu_monitor.py
```

**Funciones del Monitor:**
- 🔍 Uso de GPU en tiempo real
- 📊 Estadísticas de memoria VRAM
- 🔥 Temperatura de GPUs
- ⚡ Velocidad de procesamiento
- 📈 Progreso de documentos
- ❌ Errores y warnings

**Controles:**
- `q`: Salir
- `r`: Refresh manual
- `c`: Limpiar pantalla

### Comandos de Diagnóstico

```bash
# Verificar GPUs disponibles
nvidia-smi -L

# Monitoreo continuo de GPUs
nvidia-smi -l 1

# Verificar PyTorch + CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Verificar memoria GPU
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_properties(i).total_memory//1024**3}GB') for i in range(torch.cuda.device_count())]"
```

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"

```bash
# Reducir batch size por GPU
# En multi_gpu_config.json:
"batch_size_per_gpu": 256  # Reducir de 512
```

### GPUs no detectadas

```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Reinstalar PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Rendimiento bajo

1. **Verificar utilización GPU** (debe estar >80%)
2. **Aumentar batch_size_per_gpu** si hay VRAM disponible
3. **Verificar red**: Qdrant debe estar en red rápida
4. **Verificar CPU**: No debe estar al 100%
5. **Verificar I/O**: MongoDB debe responder rápido

### Errores de conexión Qdrant

```python
# Aumentar workers y timeout en config
"qdrant": {
    "batch_size": 500,           # Reducir si hay errores
    "connection_pool_size": 10,  # Reducir conexiones
    "timeout": 60                # Aumentar timeout
}
```

## 📈 Optimizaciones por Hardware

### Para 4x L40S (192GB VRAM total)

```json
{
  "processing": {
    "batch_size_per_gpu": 512,
    "memory_threshold": 0.85,
    "pipeline_depth": 3
  }
}
```

### Para menos VRAM (ej: 4x RTX 4090)

```json
{
  "processing": {
    "batch_size_per_gpu": 128,
    "memory_threshold": 0.75,
    "pipeline_depth": 2
  }
}
```

### Para más GPUs (ej: 8x H100)

```json
{
  "system": {
    "num_gpus": 8
  },
  "processing": {
    "batch_size_per_gpu": 1024,
    "async_upload_workers": 32
  }
}
```

## 🚨 Mejores Prácticas

### Antes de Ejecutar

1. ✅ Verificar que no hay otros procesos usando GPUs
2. ✅ Confirmar conexión estable a Qdrant y MongoDB
3. ✅ Verificar espacio en disco para logs
4. ✅ Configurar límites de temperatura GPU (opcional)

### Durante Ejecución

1. 🔍 Monitorear temperatura GPUs (<80°C recomendado)
2. 📊 Verificar utilización GPU (>80% ideal)
3. 🔄 Verificar progreso en logs
4. 💾 Monitorear espacio en disco

### Después de Ejecutar

1. 🧹 Limpiar memoria GPU: `torch.cuda.empty_cache()`
2. 📋 Revisar estadísticas finales en logs
3. ✅ Verificar datos subidos a Qdrant
4. 📊 Analizar rendimiento para futuros ajustes

## 📞 Soporte y Debugging

### Logs Detallados

```bash
# Ver logs en tiempo real
tail -f bge_m3_multi_gpu.log

# Buscar errores
grep -i error bge_m3_multi_gpu.log

# Estadísticas de rendimiento
grep -i "chunks/s\|docs/s" bge_m3_multi_gpu.log
```

### Información del Sistema

```bash
# Script de diagnóstico completo
python setup_multi_gpu.py
```

### Configuración de Desarrollo

Para desarrollo y testing:

```json
{
  "processing": {
    "batch_size_per_gpu": 32,
    "pipeline_depth": 1,
    "async_upload_workers": 4
  }
}
```

---

## 🎯 Resumen de Comandos Rápidos

```bash
# Configuración inicial
python setup_multi_gpu.py

# Ejecución estándar
python launch_multi_gpu.py

# Con monitoreo
python gpu_monitor.py  # Terminal separado

# Verificar GPUs
nvidia-smi -l 1

# Ver logs
tail -f bge_m3_multi_gpu.log
```

**¡Tu sistema 4x L40S está optimizado para procesar miles de documentos por hora!** 🚀
