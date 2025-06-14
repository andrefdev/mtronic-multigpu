"""
Monitor en tiempo real para BGE-M3 Multi-GPU Processing
Monitorea rendimiento, uso de GPU, memoria y estadísticas de procesamiento
"""

import time
import json
import psutil
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import curses
import signal
import sys

class SystemMonitor:
    """Monitor del sistema y GPUs"""
    
    def __init__(self):
        self.running = True
        self.stats = {
            "start_time": time.time(),
            "gpu_stats": [],
            "system_stats": {},
            "processing_stats": {
                "documents_processed": 0,
                "chunks_processed": 0,
                "rate_docs_per_min": 0,
                "rate_chunks_per_sec": 0,
                "estimated_completion": None
            }
        }
        
    def get_gpu_stats(self) -> List[Dict]:
        """Obtiene estadísticas de todas las GPUs"""
        try:
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpu_stats.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "temp": int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            "gpu_util": int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                            "mem_util": int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                            "mem_used": int(parts[5]),
                            "mem_total": int(parts[6]),
                            "power": float(parts[7]) if parts[7] != '[Not Supported]' else 0.0
                        })
            
            return gpu_stats
            
        except Exception as e:
            print(f"Error obteniendo stats de GPU: {e}")
            return []
    
    def get_system_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Obtener procesos Python que usan más CPU/memoria
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Ordenar por uso de CPU
            python_processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "python_processes": python_processes[:5]  # Top 5
            }
            
        except Exception as e:
            print(f"Error obteniendo stats del sistema: {e}")
            return {}
    
    def parse_log_file(self, log_file: Path) -> Dict:
        """Parsea el archivo de log para extraer estadísticas de procesamiento"""
        stats = {
            "documents_processed": 0,
            "chunks_processed": 0,
            "last_rate": 0,
            "errors": 0,
            "warnings": 0
        }
        
        if not log_file.exists():
            return stats
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Procesar últimas 1000 líneas para eficiencia
            recent_lines = lines[-1000:] if len(lines) > 1000 else lines
            
            for line in recent_lines:
                line = line.strip().lower()
                
                # Documentos procesados
                if "documento" in line and "completado" in line:
                    stats["documents_processed"] += 1
                
                # Chunks procesados (buscar patrones como "1234 chunks en")
                if "chunks en" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "chunks" and i > 0:
                                chunk_count = int(parts[i-1])
                                stats["chunks_processed"] += chunk_count
                                break
                    except (ValueError, IndexError):
                        pass
                
                # Rate (buscar "chunks/s")
                if "chunks/s" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "chunks/s" in part:
                                rate_str = part.replace("chunks/s)", "").replace("(", "")
                                stats["last_rate"] = float(rate_str)
                                break
                    except (ValueError, IndexError):
                        pass
                
                # Errores y warnings
                if "error" in line:
                    stats["errors"] += 1
                if "warning" in line:
                    stats["warnings"] += 1
            
            return stats
            
        except Exception as e:
            print(f"Error parseando log: {e}")
            return stats

class CursesMonitor:
    """Monitor con interfaz curses"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.stdscr = None
        self.log_file = Path("bge_m3_multi_gpu.log")
        
    def setup_curses(self):
        """Configura la interfaz curses"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)
        curses.start_color()
        
        # Definir colores
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Verde
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Amarillo
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)  # Rojo
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Cyan
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Magenta
    
    def cleanup_curses(self):
        """Limpia la interfaz curses"""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
    
    def draw_header(self, y: int) -> int:
        """Dibuja el encabezado"""
        title = "BGE-M3 MULTI-GPU MONITOR - 4x L40S PROCESSING"
        self.stdscr.addstr(y, 2, title, curses.color_pair(4) | curses.A_BOLD)
        y += 1
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uptime = timedelta(seconds=int(time.time() - self.monitor.stats["start_time"]))
        
        self.stdscr.addstr(y, 2, f"Tiempo: {current_time} | Uptime: {uptime}")
        y += 2
        
        return y
    
    def draw_gpu_stats(self, y: int) -> int:
        """Dibuja estadísticas de GPU"""
        self.stdscr.addstr(y, 2, "GPU STATUS (4x L40S)", curses.color_pair(1) | curses.A_BOLD)
        y += 1
        
        gpu_stats = self.monitor.get_gpu_stats()
        
        if not gpu_stats:
            self.stdscr.addstr(y, 4, "No se pudieron obtener estadísticas de GPU", curses.color_pair(3))
            return y + 2
        
        # Header
        header = f"{'ID':<3} {'Nombre':<15} {'Temp':<5} {'GPU%':<5} {'Mem%':<5} {'VRAM':<12} {'Power':<8}"
        self.stdscr.addstr(y, 4, header, curses.A_BOLD)
        y += 1
        
        total_gpu_util = 0
        total_mem_used = 0
        total_mem_total = 0
        
        for gpu in gpu_stats:
            # Determinar color basado en utilización
            util_color = curses.color_pair(1)  # Verde por defecto
            if gpu["gpu_util"] > 80:
                util_color = curses.color_pair(3)  # Rojo
            elif gpu["gpu_util"] > 50:
                util_color = curses.color_pair(2)  # Amarillo
            
            # Formatear memoria
            mem_gb = f"{gpu['mem_used']/1024:.1f}/{gpu['mem_total']/1024:.1f}GB"
            
            line = f"{gpu['id']:<3} {gpu['name'][:15]:<15} {gpu['temp']:<5} {gpu['gpu_util']:<5} {gpu['mem_util']:<5} {mem_gb:<12} {gpu['power']:<8.1f}W"
            self.stdscr.addstr(y, 4, line, util_color)
            y += 1
            
            total_gpu_util += gpu["gpu_util"]
            total_mem_used += gpu["mem_used"]
            total_mem_total += gpu["mem_total"]
        
        # Totales
        avg_gpu_util = total_gpu_util / len(gpu_stats) if gpu_stats else 0
        total_mem_percent = (total_mem_used / total_mem_total) * 100 if total_mem_total > 0 else 0
        
        self.stdscr.addstr(y, 4, f"Promedio GPU: {avg_gpu_util:.1f}% | VRAM Total: {total_mem_used/1024:.1f}/{total_mem_total/1024:.1f}GB ({total_mem_percent:.1f}%)", 
                          curses.color_pair(5))
        y += 2
        
        return y
    
    def draw_system_stats(self, y: int) -> int:
        """Dibuja estadísticas del sistema"""
        self.stdscr.addstr(y, 2, "SYSTEM STATUS", curses.color_pair(1) | curses.A_BOLD)
        y += 1
        
        sys_stats = self.monitor.get_system_stats()
        
        if not sys_stats:
            self.stdscr.addstr(y, 4, "No se pudieron obtener estadísticas del sistema", curses.color_pair(3))
            return y + 2
        
        # CPU y Memoria
        cpu_color = curses.color_pair(1)
        if sys_stats["cpu_percent"] > 80:
            cpu_color = curses.color_pair(3)
        elif sys_stats["cpu_percent"] > 60:
            cpu_color = curses.color_pair(2)
        
        mem_color = curses.color_pair(1)
        if sys_stats["memory_percent"] > 80:
            mem_color = curses.color_pair(3)
        elif sys_stats["memory_percent"] > 60:
            mem_color = curses.color_pair(2)
        
        self.stdscr.addstr(y, 4, f"CPU: {sys_stats['cpu_percent']:.1f}%", cpu_color)
        self.stdscr.addstr(y, 20, f"RAM: {sys_stats['memory_used_gb']:.1f}/{sys_stats['memory_total_gb']:.1f}GB ({sys_stats['memory_percent']:.1f}%)", mem_color)
        y += 1
        
        # Disco
        self.stdscr.addstr(y, 4, f"Disk: {sys_stats['disk_used_gb']:.1f}/{sys_stats['disk_total_gb']:.1f}GB ({sys_stats['disk_percent']:.1f}%)")
        y += 2
        
        return y
    
    def draw_processing_stats(self, y: int) -> int:
        """Dibuja estadísticas de procesamiento"""
        self.stdscr.addstr(y, 2, "PROCESSING STATUS", curses.color_pair(1) | curses.A_BOLD)
        y += 1
        
        # Parsear log para obtener estadísticas
        log_stats = self.monitor.parse_log_file(self.log_file)
        
        self.stdscr.addstr(y, 4, f"Documentos procesados: {log_stats['documents_processed']}")
        y += 1
        
        self.stdscr.addstr(y, 4, f"Chunks procesados: {log_stats['chunks_processed']:,}")
        y += 1
        
        if log_stats['last_rate'] > 0:
            rate_color = curses.color_pair(1)
            if log_stats['last_rate'] > 100:
                rate_color = curses.color_pair(1)  # Verde para alta velocidad
            elif log_stats['last_rate'] > 50:
                rate_color = curses.color_pair(2)  # Amarillo para velocidad media
            else:
                rate_color = curses.color_pair(3)  # Rojo para velocidad baja
                
            self.stdscr.addstr(y, 4, f"Velocidad actual: {log_stats['last_rate']:.1f} chunks/s", rate_color)
        else:
            self.stdscr.addstr(y, 4, "Velocidad actual: Calculando...")
        y += 1
        
        # Errores y warnings
        if log_stats['errors'] > 0:
            self.stdscr.addstr(y, 4, f"Errores: {log_stats['errors']}", curses.color_pair(3))
        else:
            self.stdscr.addstr(y, 4, "Errores: 0", curses.color_pair(1))
        
        if log_stats['warnings'] > 0:
            self.stdscr.addstr(y, 20, f"Warnings: {log_stats['warnings']}", curses.color_pair(2))
        else:
            self.stdscr.addstr(y, 20, "Warnings: 0", curses.color_pair(1))
        
        y += 2
        
        return y
    
    def draw_instructions(self, y: int) -> int:
        """Dibuja instrucciones"""
        max_y, max_x = self.stdscr.getmaxyx()
        
        instructions = [
            "Presiona 'q' para salir",
            "Presiona 'r' para refresh manual",
            "Presiona 'c' para limpiar pantalla"
        ]
        
        start_y = max_y - len(instructions) - 1
        
        for i, instruction in enumerate(instructions):
            self.stdscr.addstr(start_y + i, 2, instruction, curses.color_pair(4))
        
        return start_y
    
    def run_monitor(self):
        """Ejecuta el monitor principal"""
        try:
            self.setup_curses()
            
            while self.monitor.running:
                # Limpiar pantalla
                self.stdscr.clear()
                
                y = 1
                
                # Dibujar componentes
                y = self.draw_header(y)
                y = self.draw_gpu_stats(y)
                y = self.draw_system_stats(y)
                y = self.draw_processing_stats(y)
                self.draw_instructions(y)
                
                # Refrescar pantalla
                self.stdscr.refresh()
                
                # Verificar teclas presionadas
                key = self.stdscr.getch()
                if key == ord('q'):
                    self.monitor.running = False
                    break
                elif key == ord('c'):
                    self.stdscr.clear()
                elif key == ord('r'):
                    continue  # Refresh inmediato
                
                # Esperar antes del próximo update
                time.sleep(2)
                
        except KeyboardInterrupt:
            self.monitor.running = False
        except Exception as e:
            self.cleanup_curses()
            print(f"Error en monitor: {e}")
        finally:
            self.cleanup_curses()

def signal_handler(signum, frame):
    """Handler para señales del sistema"""
    print("\nCerrando monitor...")
    sys.exit(0)

def main():
    """Función principal"""
    # Configurar handlers de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🖥️  Iniciando BGE-M3 Multi-GPU Monitor...")
    print("Presiona Ctrl+C para salir\n")
    
    # Verificar que nvidia-smi esté disponible
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: nvidia-smi no disponible. ¿Están instalados los drivers NVIDIA?")
        sys.exit(1)
    
    # Crear y ejecutar monitor
    monitor = SystemMonitor()
    curses_monitor = CursesMonitor(monitor)
    
    try:
        curses_monitor.run_monitor()
    except Exception as e:
        print(f"Error ejecutando monitor: {e}")
        sys.exit(1)
    
    print("Monitor cerrado correctamente.")

if __name__ == "__main__":
    main()
