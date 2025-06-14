# Solución para Error de Vulnerabilidad PyTorch CVE-2025-32434

## Problema
```
ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
```

## Causa
Este error ocurre porque el modelo BGE-M3 intenta usar `torch.load` con una versión de PyTorch menor a 2.6, que tiene una vulnerabilidad de seguridad conocida (CVE-2025-32434).

## Soluciones Implementadas

### 1. Actualización de PyTorch
```bash
pip install torch>=2.6.0
```

### 2. Uso de SafeTensors (Preferido)
El código ha sido modificado para usar `safetensors` por defecto, que es más seguro:

```python
model_kwargs={
    "torch_dtype": torch.float16,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "use_safetensors": True,  # Forzar uso de safetensors
}
```

### 3. Función de Carga Segura
Se agregó la función `load_bge_model_safe()` que intenta múltiples estrategias:

1. Primero intenta con `safetensors`
2. Si falla, intenta sin `safetensors` 
3. Proporciona logging detallado de cada intento

## Instalación Recomendada

### Paso 1: Actualizar PyTorch
```bash
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements_multi_gpu.txt
```

### Paso 3: Probar la carga del modelo
```bash
python test_model_loading.py
```

## Verificación de la Instalación

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Notas Adicionales

- **Safetensors**: Es el formato preferido por su seguridad y velocidad
- **PyTorch 2.6+**: Requerido para manejar la vulnerabilidad de seguridad
- **Flash Attention**: Comentado en Windows por problemas de compilación
- **Triton**: Comentado en Windows por no estar disponible

## Troubleshooting

### Si el modelo no se descarga con safetensors:
```bash
# Forzar descarga del modelo con safetensors
huggingface-cli download BAAI/bge-m3 --cache-dir ./model_cache
```

### Si persisten errores de memoria:
```python
# Reducir batch size en la configuración
config.batch_size_per_gpu = 256  # En lugar de 512
```

### Si hay errores de CUDA:
```bash
# Verificar instalación de CUDA
nvidia-smi
nvcc --version
```
