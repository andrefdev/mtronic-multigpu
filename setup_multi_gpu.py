"""
Script de configuración y verificación del sistema para BGE-M3 Multi-GPU
Verifica hardware, instala dependencias y configura el entorno óptimo para 4x L40S
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path

def run_command(command, check=True, capture_output=True):
    """Ejecuta un comando del sistema"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando comando: {command}")
        print(f"Error: {e}")
        return None

def check_system_requirements():
    """Verifica los requisitos del sistema"""
    print("🔍 Verificando requisitos del sistema...")
    
    # Python version
    python_version = sys.version_info
    print(f"✓ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        return False
    
    # OS
    os_name = platform.system()
    print(f"✓ OS: {os_name} {platform.release()}")
    
    # CPU cores
    try:
        import psutil
        cpu_cores = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✓ CPU: {cpu_cores} cores")
        print(f"✓ RAM: {ram_gb:.1f} GB")
        
        if cpu_cores < 16:
            print("⚠️  Advertencia: Se recomiendan 16+ cores CPU para óptimo rendimiento")
        if ram_gb < 32:
            print("⚠️  Advertencia: Se recomiendan 32+ GB RAM para óptimo rendimiento")
            
    except ImportError:
        print("⚠️  psutil no disponible, no se puede verificar CPU/RAM")
    
    return True

def check_cuda_installation():
    """Verifica la instalación de CUDA"""
    print("\n🚀 Verificando CUDA...")
    
    # NVIDIA-SMI
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    
    if nvidia_smi is None:
        print("❌ Error: nvidia-smi no encontrado. ¿Está instalado el driver NVIDIA?")
        return False
    
    # Parsear información de GPUs
    gpu_lines = nvidia_smi.strip().split('\n')
    gpus = []
    
    for i, line in enumerate(gpu_lines):
        if line.strip():
            parts = line.split(', ')
            if len(parts) >= 2:
                name = parts[0].strip()
                memory = parts[1].strip()
                gpus.append({"id": i, "name": name, "memory_gb": int(memory) // 1024})
                print(f"✓ GPU {i}: {name} - {int(memory) // 1024} GB VRAM")
    
    if len(gpus) < 4:
        print(f"⚠️  Solo se detectaron {len(gpus)} GPUs. Se esperaban 4x L40S")
    
    # Verificar si son L40S
    l40s_count = sum(1 for gpu in gpus if "L40S" in gpu["name"])
    if l40s_count > 0:
        print(f"✓ Detectadas {l40s_count} GPUs L40S")
    else:
        print("⚠️  No se detectaron GPUs L40S específicamente")
    
    # Verificar CUDA version
    cuda_version = run_command("nvcc --version")
    if cuda_version:
        print(f"✓ CUDA instalado: {cuda_version.split('release ')[-1].split(',')[0] if 'release' in cuda_version else 'versión no detectada'}")
    else:
        print("⚠️  NVCC no encontrado, pero las GPUs están disponibles")
    
    return len(gpus) > 0

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("\n📦 Instalando dependencias...")
    
    requirements_file = Path(__file__).parent / "requirements_multi_gpu.txt"
    
    if not requirements_file.exists():
        print(f"❌ Error: No se encontró {requirements_file}")
        return False
    
    # Actualizar pip
    print("Actualizando pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", capture_output=False)
    
    # Instalar PyTorch con CUDA
    print("Instalando PyTorch con soporte CUDA...")
    torch_install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    run_command(torch_install_cmd, capture_output=False)
    
    # Instalar resto de dependencias
    print(f"Instalando dependencias desde {requirements_file}...")
    pip_install_cmd = f"{sys.executable} -m pip install -r {requirements_file}"
    result = run_command(pip_install_cmd, capture_output=False, check=False)
    
    if result is None:
        print("⚠️  Algunas dependencias pueden no haberse instalado correctamente")
    
    return True

def verify_pytorch_cuda():
    """Verifica que PyTorch detecte CUDA"""
    print("\n🔥 Verificando PyTorch + CUDA...")
    
    try:
        import torch
        
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPUs detectadas por PyTorch: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} - {gpu_memory:.1f} GB")
            
            # Test básico
            print("Ejecutando test básico de CUDA...")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"✓ Test CUDA exitoso: {z.shape}")
            
            return True
        else:
            print("❌ Error: PyTorch no detecta CUDA")
            return False
            
    except ImportError:
        print("❌ Error: PyTorch no instalado")
        return False
    except Exception as e:
        print(f"❌ Error en test de PyTorch: {e}")
        return False

def create_config_file():
    """Crea archivo de configuración optimizado"""
    print("\n⚙️  Creando configuración optimizada...")
    
    config = {
        "system": {
            "num_gpus": 4,
            "gpu_type": "L40S",
            "total_vram_gb": 192,  # 4 x 48GB
            "cpu_cores": 32,
            "optimization_level": "maximum"
        },
        "model": {
            "name": "BAAI/bge-m3",
            "precision": "fp16",
            "max_length": 8192,
            "use_flash_attention": True
        },
        "processing": {
            "batch_size_per_gpu": 512,
            "num_workers_per_gpu": 4,
            "async_upload_workers": 16,
            "pipeline_depth": 3,
            "memory_threshold": 0.85
        },
        "qdrant": {
            "batch_size": 1000,
            "connection_pool_size": 20,
            "timeout": 30
        },
        "mongodb": {
            "connection_pool_size": 10,
            "max_idle_time": 30000
        },
        "monitoring": {
            "log_level": "INFO",
            "memory_monitoring": True,
            "performance_tracking": True,
            "gpu_stats_interval": 30
        }
    }
    
    config_file = Path(__file__).parent / "multi_gpu_config.json"
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ Configuración guardada en: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error creando configuración: {e}")
        return False

def setup_environment_variables():
    """Configura variables de entorno optimizadas"""
    print("\n🌍 Configurando variables de entorno...")
    
    env_vars = {
        "CUDA_LAUNCH_BLOCKING": "0",
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",  # Menos verbose que INFO
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "8",  # 32 cores / 4 GPUs
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_CUDNN_V8_API_DISABLED": "1"  # Para estabilidad
    }
    
    env_file = Path(__file__).parent / ".env.multi_gpu"
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
                print(f"  {key}={value}")
        
        print(f"✓ Variables de entorno guardadas en: {env_file}")
        print("💡 Tip: Cargar con 'source .env.multi_gpu' o usar python-dotenv")
        return True
        
    except Exception as e:
        print(f"❌ Error creando archivo de entorno: {e}")
        return False

def create_launcher_script():
    """Crea script de lanzamiento optimizado"""
    print("\n🚀 Creando script de lanzamiento...")
    
    launcher_content = '''#!/usr/bin/env python3
"""
Launcher optimizado para BGE-M3 Multi-GPU Processing
Configura el entorno y ejecuta el procesamiento con 4x L40S GPUs
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Configura el entorno antes de ejecutar"""
    # Cargar variables de entorno
    env_file = Path(__file__).parent / ".env.multi_gpu"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("✓ Variables de entorno cargadas")
    
    # Verificar CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Error: CUDA no disponible")
            sys.exit(1)
        
        gpu_count = torch.cuda.device_count()
        print(f"✓ {gpu_count} GPUs disponibles")
        
        if gpu_count < 4:
            print(f"⚠️  Solo {gpu_count} GPUs detectadas, se esperaban 4")
        
    except ImportError:
        print("❌ Error: PyTorch no disponible")
        sys.exit(1)

def main():
    """Función principal del launcher"""
    print("🚀 BGE-M3 Multi-GPU Launcher")
    print("=" * 50)
    
    setup_environment()
    
    # Ejecutar script principal
    script_path = Path(__file__).parent / "bge_m3_multi_gpu_optimized.py"
    
    if not script_path.exists():
        print(f"❌ Error: Script no encontrado: {script_path}")
        sys.exit(1)
    
    print(f"Ejecutando: {script_path}")
    
    try:
        # Ejecutar con optimizaciones
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "CUDA_LAUNCH_BLOCKING": "0"
        })
        
        subprocess.run([sys.executable, str(script_path)], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\\n⏹️  Proceso interrumpido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando script: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    launcher_file = Path(__file__).parent / "launch_multi_gpu.py"
    
    try:
        with open(launcher_file, 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        
        # Hacer ejecutable en Unix
        if os.name != 'nt':
            os.chmod(launcher_file, 0o755)
        
        print(f"✓ Launcher creado: {launcher_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creando launcher: {e}")
        return False

def main():
    """Función principal de configuración"""
    print("🔧 BGE-M3 Multi-GPU Setup & Configuration")
    print("=" * 60)
    print("Configurando entorno para 4x L40S GPUs con máximo rendimiento")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 7
    
    # 1. Verificar sistema
    if check_system_requirements():
        success_steps += 1
    
    # 2. Verificar CUDA
    if check_cuda_installation():
        success_steps += 1
    
    # 3. Instalar dependencias
    if install_dependencies():
        success_steps += 1
    
    # 4. Verificar PyTorch
    if verify_pytorch_cuda():
        success_steps += 1
    
    # 5. Crear configuración
    if create_config_file():
        success_steps += 1
    
    # 6. Variables de entorno
    if setup_environment_variables():
        success_steps += 1
    
    # 7. Script de lanzamiento
    if create_launcher_script():
        success_steps += 1
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE CONFIGURACIÓN")
    print("=" * 60)
    print(f"✅ Pasos completados: {success_steps}/{total_steps}")
    
    if success_steps == total_steps:
        print("🎉 ¡Configuración completada exitosamente!")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Ejecutar: python launch_multi_gpu.py")
        print("2. Monitorear logs en tiempo real")
        print("3. Verificar uso de GPU con: nvidia-smi -l 1")
        print("\n💡 TIPS DE RENDIMIENTO:")
        print("- Asegurar que no hay otros procesos usando las GPUs")
        print("- Monitorear temperatura de GPUs durante procesamiento intensivo")
        print("- Considerar ajustar batch_size según el dataset específico")
    else:
        print("⚠️  Configuración parcialmente completada")
        print("Revisar errores anteriores antes de continuar")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
