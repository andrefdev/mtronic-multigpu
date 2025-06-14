"""
Script de prueba rápida para verificar el sistema Multi-GPU BGE-M3
Ejecuta tests básicos para confirmar que todo está configurado correctamente
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Verifica que todas las dependencias estén disponibles"""
    print("🔍 Verificando imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        from FlagEmbedding import BGEM3FlagModel
        print("✅ FlagEmbedding: Disponible")
        
        import psutil
        print(f"✅ PSUtil: {psutil.__version__}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False

def test_cuda():
    """Verifica CUDA y GPUs"""
    print("\n🚀 Verificando CUDA y GPUs...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPUs detectadas: {gpu_count}")
        
        if gpu_count < 4:
            print(f"⚠️  Solo {gpu_count} GPUs detectadas, se esperaban 4")
        
        # Información detallada de cada GPU
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} - {memory_gb:.1f}GB VRAM")
            
            # Verificar que es L40S
            if "L40S" in props.name:
                print(f"    ✅ L40S detectada")
            else:
                print(f"    ⚠️  No es L40S: {props.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando CUDA: {e}")
        return False

def test_memory_allocation():
    """Test básico de asignación de memoria GPU"""
    print("\n💾 Verificando asignación de memoria GPU...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible para test de memoria")
            return False
        
        gpu_count = torch.cuda.device_count()
        
        for gpu_id in range(min(gpu_count, 4)):
            torch.cuda.set_device(gpu_id)
            
            # Asignar tensor pequeño
            x = torch.randn(1000, 1000, device=f"cuda:{gpu_id}", dtype=torch.float16)
            y = torch.randn(1000, 1000, device=f"cuda:{gpu_id}", dtype=torch.float16)
            
            # Operación simple
            z = torch.mm(x, y)
            
            # Verificar memoria
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            
            print(f"  GPU {gpu_id}: Test exitoso - {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Limpiar
            del x, y, z
            torch.cuda.empty_cache()
        
        print("✅ Test de memoria GPU exitoso")
        return True
        
    except Exception as e:
        print(f"❌ Error en test de memoria: {e}")
        return False

def test_bge_model():
    """Test de carga del modelo BGE-M3"""
    print("\n🤖 Verificando carga del modelo BGE-M3...")
    
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible para test de modelo")
            return False
        
        print("  Cargando modelo BGE-M3...")
        start_time = time.time()
        
        model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            device="cuda:0",
            model_kwargs={
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        )
        
        load_time = time.time() - start_time
        print(f"  ✅ Modelo cargado en {load_time:.1f}s")
        
        # Test de embedding simple
        test_texts = ["Este es un texto de prueba.", "Another test sentence."]
        
        print("  Generando embeddings de prueba...")
        start_time = time.time()
        
        embeddings = model.encode(
            sentences=test_texts,
            batch_size=2,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        
        inference_time = time.time() - start_time
        print(f"  ✅ Embeddings generados en {inference_time:.3f}s")
        print(f"  📊 Shape: {embeddings.shape}")
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo BGE-M3: {e}")
        traceback.print_exc()
        return False

def test_multi_gpu_allocation():
    """Test de asignación en múltiples GPUs"""
    print("\n🔄 Verificando asignación multi-GPU...")
    
    try:
        import torch
        import threading
        import queue
        
        if not torch.cuda.is_available():
            print("❌ CUDA no disponible")
            return False
        
        gpu_count = torch.cuda.device_count()
        test_gpus = min(gpu_count, 4)
        
        def gpu_worker(gpu_id, result_queue):
            try:
                torch.cuda.set_device(gpu_id)
                
                # Operación en GPU
                x = torch.randn(2000, 2000, device=f"cuda:{gpu_id}", dtype=torch.float16)
                y = torch.randn(2000, 2000, device=f"cuda:{gpu_id}", dtype=torch.float16)
                
                start_time = time.time()
                z = torch.mm(x, y)
                gpu_time = time.time() - start_time
                
                # Verificar resultado
                result_sum = z.sum().item()
                
                result_queue.put({
                    "gpu_id": gpu_id,
                    "success": True,
                    "time": gpu_time,
                    "result_sum": result_sum
                })
                
                # Limpiar
                del x, y, z
                torch.cuda.empty_cache()
                
            except Exception as e:
                result_queue.put({
                    "gpu_id": gpu_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Ejecutar en paralelo
        result_queue = queue.Queue()
        threads = []
        
        start_time = time.time()
        
        for gpu_id in range(test_gpus):
            thread = threading.Thread(target=gpu_worker, args=(gpu_id, result_queue))
            threads.append(thread)
            thread.start()
        
        # Esperar resultados
        for thread in threads:
            thread.join(timeout=30)
        
        total_time = time.time() - start_time
        
        # Recoger resultados
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Analizar resultados
        successful_gpus = [r for r in results if r["success"]]
        failed_gpus = [r for r in results if not r["success"]]
        
        print(f"  ✅ GPUs exitosas: {len(successful_gpus)}/{test_gpus}")
        print(f"  ⏱️  Tiempo total: {total_time:.2f}s")
        
        for result in successful_gpus:
            print(f"    GPU {result['gpu_id']}: {result['time']:.3f}s")
        
        if failed_gpus:
            print("  ❌ GPUs con errores:")
            for result in failed_gpus:
                print(f"    GPU {result['gpu_id']}: {result['error']}")
        
        return len(successful_gpus) == test_gpus
        
    except Exception as e:
        print(f"❌ Error en test multi-GPU: {e}")
        return False

def test_async_capabilities():
    """Test básico de capacidades asíncronas"""
    print("\n🔄 Verificando capacidades asíncronas...")
    
    try:
        import asyncio
        import aiohttp
        
        async def simple_async_test():
            # Test básico de asyncio
            await asyncio.sleep(0.1)
            return "async_ok"
        
        # Ejecutar test
        result = asyncio.run(simple_async_test())
        
        if result == "async_ok":
            print("  ✅ AsyncIO funcionando correctamente")
            return True
        else:
            print("  ❌ Error en AsyncIO")
            return False
            
    except Exception as e:
        print(f"❌ Error en test async: {e}")
        return False

def test_configuration_files():
    """Verifica que los archivos de configuración existan"""
    print("\n📁 Verificando archivos de configuración...")
    
    files_to_check = [
        "bge_m3_multi_gpu_optimized.py",
        "requirements_multi_gpu.txt",
        "setup_multi_gpu.py",
        "gpu_monitor.py",
        "README_multi_gpu.md"
    ]
    
    current_dir = Path(__file__).parent
    all_exist = True
    
    for filename in files_to_check:
        file_path = current_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  ✅ {filename} - {size_kb:.1f}KB")
        else:
            print(f"  ❌ {filename} - NO ENCONTRADO")
            all_exist = False
    
    # Verificar archivos opcionales
    optional_files = [
        "multi_gpu_config.json",
        ".env.multi_gpu",
        "launch_multi_gpu.py"
    ]
    
    print("\n  📋 Archivos opcionales (se crean automáticamente):")
    for filename in optional_files:
        file_path = current_dir / filename
        if file_path.exists():
            print(f"    ✅ {filename}")
        else:
            print(f"    ⚠️  {filename} - Será creado por setup_multi_gpu.py")
    
    return all_exist

def main():
    """Función principal de tests"""
    print("🧪 BGE-M3 MULTI-GPU SYSTEM TEST")
    print("=" * 60)
    print("Verificando configuración para 4x L40S GPUs...")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA y GPUs", test_cuda),
        ("Memoria GPU", test_memory_allocation),
        ("Modelo BGE-M3", test_bge_model),
        ("Multi-GPU", test_multi_gpu_allocation),
        ("Async", test_async_capabilities),
        ("Archivos", test_configuration_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLIDO")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        print()  # Línea en blanco entre tests
    
    # Resumen final
    print("=" * 60)
    print("📊 RESUMEN DE TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests exitosos ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS EXITOSOS!")
        print("🚀 El sistema está listo para procesamiento Multi-GPU")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Ejecutar: python launch_multi_gpu.py")
        print("2. Monitorear: python gpu_monitor.py (terminal separado)")
        print("3. Verificar logs: tail -f bge_m3_multi_gpu.log")
    else:
        failed = total - passed
        print(f"\n⚠️  {failed} tests fallaron")
        print("🔧 Revisar errores antes de continuar")
        print("💡 Ejecutar: python setup_multi_gpu.py para configuración automática")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
