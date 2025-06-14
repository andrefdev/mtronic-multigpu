#!/usr/bin/env python3
"""
Script de prueba para verificar la carga del modelo BGE-M3 con manejo de errores
"""

import torch
import logging
from FlagEmbedding import BGEM3FlagModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Prueba la carga del modelo BGE-M3 con diferentes configuraciones"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA no está disponible")
        return False
    
    device = "cuda:0"
    model_name = "BAAI/bge-m3"
    
    # Intentar con safetensors primero
    try:
        logger.info("Intentando cargar modelo con safetensors...")
        model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            device=device,
            model_kwargs={
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            },
        )
        logger.info("✅ Modelo cargado exitosamente con safetensors")
        return True
    except Exception as e:
        logger.warning(f"❌ Error cargando con safetensors: {e}")
    
    # Fallback sin safetensors
    try:
        logger.info("Intentando cargar modelo sin safetensors...")
        model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            device=device,
            model_kwargs={
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        )
        logger.info("✅ Modelo cargado exitosamente sin safetensors")
        return True
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando prueba de carga del modelo BGE-M3...")
    
    # Verificar versión de PyTorch
    logger.info(f"Versión de PyTorch: {torch.__version__}")
    logger.info(f"Versión de CUDA: {torch.version.cuda}")
    logger.info(f"GPUs disponibles: {torch.cuda.device_count()}")
    
    success = test_model_loading()
    
    if success:
        logger.info("🎉 Prueba completada exitosamente")
    else:
        logger.error("❌ Prueba fallida")
