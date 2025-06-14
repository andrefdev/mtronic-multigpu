"""
BGE-M3 Multi-GPU Optimized Script para 4x L40S GPUs
Máximo rendimiento para generación de embeddings y almacenamiento en Qdrant
"""

from FlagEmbedding import BGEM3FlagModel
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Dict, List, Tuple, Optional
from QdrantService import QdrantService
from MongoDBService import MongoDBService
from dotenv import load_dotenv
import asyncio
import aiohttp
import time
import gc
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import threading
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import GPUtil

load_dotenv()

# Configuración optimizada para 4x L40S
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["OMP_NUM_THREADS"] = "8"  # 32 cores / 4 GPUs

@dataclass
class ProcessingConfig:
    """Configuración optimizada para 4x L40S GPUs"""
    num_gpus: int = 4
    batch_size_per_gpu: int = 512  # L40S tiene 48GB VRAM
    max_length: int = 8192
    num_workers_per_gpu: int = 4
    async_upload_workers: int = 16
    prefetch_factor: int = 4
    cpu_cores: int = 32
    memory_threshold: float = 0.85  # 85% memoria GPU
    qdrant_batch_size: int = 1000
    pipeline_depth: int = 3  # Pipeline stages

class GPUMemoryManager:
    """Gestor avanzado de memoria GPU"""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
    @contextmanager
    def memory_context(self):
        """Context manager para gestión automática de memoria"""
        try:
            torch.cuda.set_device(self.gpu_id)
            initial_memory = torch.cuda.memory_allocated(self.gpu_id)
            yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            final_memory = torch.cuda.memory_allocated(self.gpu_id)
            freed = initial_memory - final_memory
            if freed > 0:
                logging.debug(f"GPU {self.gpu_id}: Liberados {freed / 1024**3:.2f}GB")
    
    def get_memory_stats(self) -> Dict:
        """Obtiene estadísticas detalladas de memoria"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
            return {
                "gpu_id": self.gpu_id,
                "allocated": allocated,
                "reserved": reserved,
                "total": total,
                "free": total - reserved,
                "usage_percent": (reserved / total) * 100
            }
        return {}

class AsyncQdrantUploader:
    """Uploader asíncrono optimizado para Qdrant"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.upload_queue = asyncio.Queue(maxsize=10000)
        self.stats = {"uploaded": 0, "failed": 0, "total_time": 0}
        
    async def upload_worker(self, worker_id: int):
        """Worker asíncrono para subir embeddings"""
        qdrant = QdrantService()
        batch = []
        
        while True:
            try:
                # Recoger batch
                for _ in range(self.config.qdrant_batch_size):
                    try:
                        item = await asyncio.wait_for(
                            self.upload_queue.get(), timeout=1.0
                        )
                        if item is None:  # Señal de finalización
                            break
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Subir batch
                start_time = time.time()
                success_count = await self._upload_batch(qdrant, batch)
                upload_time = time.time() - start_time
                
                self.stats["uploaded"] += success_count
                self.stats["failed"] += len(batch) - success_count
                self.stats["total_time"] += upload_time
                
                logging.info(
                    f"Worker {worker_id}: Subidos {success_count}/{len(batch)} "
                    f"embeddings en {upload_time:.2f}s"
                )
                
                batch.clear()
                
            except Exception as e:
                logging.error(f"Error en upload worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _upload_batch(self, qdrant: QdrantService, batch: List[Tuple]) -> int:
        """Sube un batch de embeddings de forma asíncrona"""
        success_count = 0
        
        # Preparar datos para inserción batch
        points_data = []
        for content, embedding, metadata in batch:
            try:
                point_data = qdrant.prepare_point_data(content, embedding, metadata)
                points_data.append(point_data)
                success_count += 1
            except Exception as e:
                logging.error(f"Error preparando punto: {e}")
        
        # Inserción batch en Qdrant
        if points_data:
            try:
                await qdrant.batch_upload_async(points_data)
            except Exception as e:
                logging.error(f"Error en batch upload: {e}")
                success_count = 0
        
        return success_count
    
    async def add_to_queue(self, content: str, embedding: np.ndarray, metadata: Dict):
        """Añade un embedding a la cola de subida"""
        await self.upload_queue.put((content, embedding, metadata))
    
    async def start_workers(self):
        """Inicia los workers de subida"""
        tasks = []
        for i in range(self.config.async_upload_workers):
            task = asyncio.create_task(self.upload_worker(i))
            tasks.append(task)
        return tasks
    
    async def stop_workers(self):
        """Detiene los workers de subida"""
        # Enviar señales de finalización
        for _ in range(self.config.async_upload_workers):
            await self.upload_queue.put(None)

def load_bge_model_safe(model_name: str, gpu_id: int, use_fp16: bool = True) -> BGEM3FlagModel:
    """
    Carga el modelo BGE-M3 de manera segura manejando errores de versión de PyTorch
    """
    device = f"cuda:{gpu_id}"
    
    # Intentar con safetensors primero
    try:
        logging.info(f"Intentando cargar modelo con safetensors en GPU {gpu_id}")
        model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
            model_kwargs={
                "torch_dtype": torch.float16 if use_fp16 else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            },
        )
        logging.info(f"Modelo cargado exitosamente con safetensors en GPU {gpu_id}")
        return model
    except Exception as e:
        logging.warning(f"Error cargando con safetensors en GPU {gpu_id}: {e}")
    
    # Fallback sin safetensors
    try:
        logging.info(f"Intentando cargar modelo sin safetensors en GPU {gpu_id}")
        model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
            model_kwargs={
                "torch_dtype": torch.float16 if use_fp16 else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        )
        logging.info(f"Modelo cargado exitosamente sin safetensors en GPU {gpu_id}")
        return model
    except Exception as e:
        logging.error(f"Error cargando modelo en GPU {gpu_id}: {e}")
        raise

class MultiGPUEmbeddingGenerator:
    """Generador de embeddings optimizado para múltiples GPUs"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.models = {}
        self.memory_managers = {}
        self.setup_gpus()
        
    def setup_gpus(self):
        """Configura y prepara todas las GPUs"""
        logging.info(f"Configurando {self.config.num_gpus} GPUs L40S...")
        
        for gpu_id in range(self.config.num_gpus):
            self.memory_managers[gpu_id] = GPUMemoryManager(gpu_id)
            
            # Configurar GPU
            torch.cuda.set_device(gpu_id)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Cargar modelo en GPU con manejo seguro
            with self.memory_managers[gpu_id].memory_context():
                try:
                    model = load_bge_model_safe(
                        "BAAI/bge-m3",
                        gpu_id,
                        use_fp16=True
                    )
                    self.models[gpu_id] = model
                except Exception as e:
                    logging.error(f"Error cargando modelo en GPU {gpu_id}: {e}")
                    continue
                
            logging.info(f"GPU {gpu_id} configurada correctamente")
            
        logging.info("Todas las GPUs configuradas exitosamente")
    
    def generate_embeddings_gpu(
        self, 
        gpu_id: int, 
        texts: List[str], 
        result_queue: queue.Queue
    ):
        """Genera embeddings en una GPU específica"""
        try:
            torch.cuda.set_device(gpu_id)
            model = self.models[gpu_id]
            memory_manager = self.memory_managers[gpu_id]
            
            with memory_manager.memory_context():
                # Procesar en batches
                batch_size = self.config.batch_size_per_gpu
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    start_time = time.time()
                    
                    # Verificar memoria antes del procesamiento
                    memory_stats = memory_manager.get_memory_stats()
                    if memory_stats["usage_percent"] > (self.config.memory_threshold * 100):
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Generar embeddings
                    embeddings = model.encode(
                        sentences=batch_texts,
                        batch_size=min(256, len(batch_texts)),
                        max_length=self.config.max_length,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )["dense_vecs"]
                    
                    processing_time = time.time() - start_time
                    
                    # Enviar resultados
                    result_queue.put({
                        "gpu_id": gpu_id,
                        "batch_start": i,
                        "embeddings": embeddings,
                        "processing_time": processing_time,
                        "batch_size": len(batch_texts)
                    })
                    
                    logging.debug(
                        f"GPU {gpu_id}: Batch {i//batch_size + 1} procesado "
                        f"({len(batch_texts)} textos en {processing_time:.2f}s)"
                    )
                    
        except Exception as e:
            logging.error(f"Error en GPU {gpu_id}: {e}")
            result_queue.put({"error": str(e), "gpu_id": gpu_id})
    
    def generate_embeddings_parallel(self, texts: List[str]) -> List[np.ndarray]:
        """Genera embeddings usando todas las GPUs en paralelo"""
        if not texts:
            return []
        
        # Dividir textos entre GPUs
        texts_per_gpu = len(texts) // self.config.num_gpus
        text_chunks = []
        
        for i in range(self.config.num_gpus):
            start_idx = i * texts_per_gpu
            if i == self.config.num_gpus - 1:  # Última GPU toma el resto
                end_idx = len(texts)
            else:
                end_idx = (i + 1) * texts_per_gpu
            text_chunks.append(texts[start_idx:end_idx])
        
        # Cola para resultados
        result_queue = queue.Queue()
        
        # Crear threads para cada GPU
        threads = []
        for gpu_id, text_chunk in enumerate(text_chunks):
            if text_chunk:  # Solo si hay textos para procesar
                thread = threading.Thread(
                    target=self.generate_embeddings_gpu,
                    args=(gpu_id, text_chunk, result_queue)
                )
                threads.append(thread)
                thread.start()
        
        # Recoger resultados
        all_embeddings = [None] * len(texts)
        completed_gpus = 0
        
        while completed_gpus < len(threads):
            try:
                result = result_queue.get(timeout=60)  # 60s timeout
                
                if "error" in result:
                    logging.error(f"Error en GPU {result['gpu_id']}: {result['error']}")
                    completed_gpus += 1
                    continue
                
                # Procesar resultado exitoso
                gpu_id = result["gpu_id"]
                batch_start = result["batch_start"]
                embeddings = result["embeddings"]
                
                # Calcular índices globales
                global_start = gpu_id * texts_per_gpu + batch_start
                for j, embedding in enumerate(embeddings):
                    all_embeddings[global_start + j] = embedding
                
                # Verificar si la GPU completó todos sus batches
                texts_for_gpu = len(text_chunks[gpu_id])
                if batch_start + len(embeddings) >= texts_for_gpu:
                    completed_gpus += 1
                    
            except queue.Empty:
                logging.warning("Timeout esperando resultados de GPU")
                break
        
        # Esperar a que terminen todos los threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Filtrar None values
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        
        if len(valid_embeddings) != len(texts):
            logging.warning(
                f"Se generaron {len(valid_embeddings)} embeddings "
                f"de {len(texts)} textos esperados"
            )
        
        return valid_embeddings
    
    def get_all_gpu_stats(self) -> Dict:
        """Obtiene estadísticas de todas las GPUs"""
        stats = {}
        for gpu_id in range(self.config.num_gpus):
            stats[f"gpu_{gpu_id}"] = self.memory_managers[gpu_id].get_memory_stats()
        return stats

class OptimizedBgeService:
    """Servicio principal optimizado para 4x L40S GPUs"""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self.embedding_generator = MultiGPUEmbeddingGenerator(self.config)
        self.uploader = AsyncQdrantUploader(self.config)
        self.mongo = MongoDBService()
        self.qdrant = QdrantService()
        
        # Estadísticas
        self.stats = {
            "documents_processed": 0,
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "upload_success": 0,
            "upload_failed": 0,
            "total_time": 0,
            "gpu_time": 0,
            "upload_time": 0
        }
        
        self.logger.info("Servicio optimizado inicializado para 4x L40S GPUs")
    
    async def process_document_pipeline(
        self, 
        uuid: str, 
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Procesa un documento usando pipeline optimizado"""
        async with semaphore:
            try:
                start_time = time.time()
                
                # Verificar si ya existe
                if self.qdrant.check_pdf_exists(uuid):
                    self.logger.info(f"Documento {uuid} ya procesado")
                    return True
                
                # Obtener chunks
                chunks = self.mongo.get_chunks_by_uuid(uuid)
                if not chunks:
                    self.logger.error(f"No se encontraron chunks para {uuid}")
                    return False
                
                self.logger.info(f"Procesando {len(chunks)} chunks de {uuid}")
                
                # Preparar textos
                texts = [f"search_document: {chunk['content']}" for chunk in chunks]
                
                # Generar embeddings en paralelo (todas las GPUs)
                gpu_start = time.time()
                embeddings = self.embedding_generator.generate_embeddings_parallel(texts)
                gpu_time = time.time() - gpu_start
                
                if len(embeddings) != len(chunks):
                    self.logger.error(
                        f"Mismatch: {len(embeddings)} embeddings para "
                        f"{len(chunks)} chunks en {uuid}"
                    )
                    return False
                
                # Subir embeddings de forma asíncrona
                upload_start = time.time()
                upload_tasks = []
                
                for chunk, embedding in zip(chunks, embeddings):
                    task = self.uploader.add_to_queue(
                        chunk["content"], 
                        embedding, 
                        chunk["metadata"]
                    )
                    upload_tasks.append(task)
                
                # Esperar a que se añadan todas las tareas
                await asyncio.gather(*upload_tasks)
                upload_time = time.time() - upload_start
                
                # Actualizar estadísticas
                processing_time = time.time() - start_time
                self.stats["documents_processed"] += 1
                self.stats["chunks_processed"] += len(chunks)
                self.stats["embeddings_generated"] += len(embeddings)
                self.stats["gpu_time"] += gpu_time
                self.stats["upload_time"] += upload_time
                self.stats["total_time"] += processing_time
                
                chunks_per_second = len(chunks) / processing_time
                
                self.logger.info(
                    f"Documento {uuid} completado: {len(chunks)} chunks en "
                    f"{processing_time:.2f}s ({chunks_per_second:.1f} chunks/s) "
                    f"[GPU: {gpu_time:.2f}s, Upload: {upload_time:.2f}s]"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error procesando documento {uuid}: {e}")
                return False
    
    async def process_all_documents(self):
        """Procesa todos los documentos con máximo paralelismo"""
        try:
            # Obtener UUIDs
            uuids = self.mongo.get_uuids_sorted_by_insertion()
            total_docs = len(uuids)
            
            if not uuids:
                self.logger.info("No hay documentos para procesar")
                return
            
            self.logger.info(f"Procesando {total_docs} documentos con 4x L40S GPUs")
            
            # Iniciar workers de upload
            upload_tasks = await self.uploader.start_workers()
            
            # Semáforo para controlar concurrencia (pipeline depth)
            semaphore = asyncio.Semaphore(self.config.pipeline_depth)
            
            # Procesar documentos
            start_time = time.time()
            
            document_tasks = [
                self.process_document_pipeline(uuid, semaphore) 
                for uuid in uuids
            ]
            
            # Procesar con seguimiento de progreso
            completed = 0
            successful = 0
            
            for task in asyncio.as_completed(document_tasks):
                result = await task
                completed += 1
                if result:
                    successful += 1
                
                # Mostrar progreso cada 10 documentos
                if completed % 10 == 0 or completed == total_docs:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_docs - completed) / rate if rate > 0 else 0
                    
                    gpu_stats = self.embedding_generator.get_all_gpu_stats()
                    avg_gpu_usage = np.mean([
                        stats["usage_percent"] 
                        for stats in gpu_stats.values()
                    ])
                    
                    self.logger.info(
                        f"Progreso: {completed}/{total_docs} "
                        f"({completed/total_docs*100:.1f}%) - "
                        f"Exitosos: {successful} - "
                        f"Rate: {rate:.1f} docs/s - "
                        f"ETA: {eta/60:.1f}min - "
                        f"GPU promedio: {avg_gpu_usage:.1f}%"
                    )
            
            # Detener workers de upload
            await self.uploader.stop_workers()
            
            # Esperar a que terminen los workers
            await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Estadísticas finales
            total_time = time.time() - start_time
            self._log_final_statistics(total_time, successful, total_docs)
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento general: {e}")
            raise
        finally:
            # Limpiar recursos
            self._cleanup_resources()
    
    def _log_final_statistics(self, total_time: float, successful: int, total_docs: int):
        """Registra estadísticas finales detalladas"""
        # Estadísticas de rendimiento
        docs_per_second = successful / total_time if total_time > 0 else 0
        chunks_per_second = self.stats["chunks_processed"] / total_time if total_time > 0 else 0
        
        # Estadísticas de GPU
        gpu_stats = self.embedding_generator.get_all_gpu_stats()
        total_vram = sum(stats["total"] for stats in gpu_stats.values())
        used_vram = sum(stats["reserved"] for stats in gpu_stats.values())
        
        # Estadísticas de upload
        upload_stats = self.uploader.stats
        upload_success_rate = (
            upload_stats["uploaded"] / 
            (upload_stats["uploaded"] + upload_stats["failed"]) * 100
            if (upload_stats["uploaded"] + upload_stats["failed"]) > 0 else 0
        )
        
        self.logger.info("=" * 80)
        self.logger.info("ESTADÍSTICAS FINALES - 4x L40S PROCESSING")
        self.logger.info("=" * 80)
        self.logger.info(f"Documentos procesados: {successful}/{total_docs} ({successful/total_docs*100:.1f}%)")
        self.logger.info(f"Chunks procesados: {self.stats['chunks_processed']:,}")
        self.logger.info(f"Embeddings generados: {self.stats['embeddings_generated']:,}")
        self.logger.info(f"Tiempo total: {total_time/60:.1f} minutos")
        self.logger.info(f"Velocidad: {docs_per_second:.1f} docs/s, {chunks_per_second:.1f} chunks/s")
        self.logger.info("-" * 40)
        self.logger.info(f"Tiempo GPU: {self.stats['gpu_time']/60:.1f}min ({self.stats['gpu_time']/total_time*100:.1f}%)")
        self.logger.info(f"Tiempo Upload: {self.stats['upload_time']/60:.1f}min ({self.stats['upload_time']/total_time*100:.1f}%)")
        self.logger.info("-" * 40)
        self.logger.info(f"VRAM total: {total_vram:.1f}GB")
        self.logger.info(f"VRAM utilizada: {used_vram:.1f}GB ({used_vram/total_vram*100:.1f}%)")
        self.logger.info("-" * 40)
        self.logger.info(f"Uploads exitosos: {upload_stats['uploaded']:,}")
        self.logger.info(f"Uploads fallidos: {upload_stats['failed']:,}")
        self.logger.info(f"Tasa éxito upload: {upload_success_rate:.1f}%")
        self.logger.info("=" * 80)
    
    def _cleanup_resources(self):
        """Limpia todos los recursos"""
        try:
            # Limpiar memoria GPU
            for gpu_id in range(self.config.num_gpus):
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            gc.collect()
            
            # Cerrar conexiones
            if hasattr(self.qdrant, 'client'):
                self.qdrant.client.close()
            if hasattr(self.mongo, 'close'):
                self.mongo.close()
                
            self.logger.info("Recursos limpiados correctamente")
            
        except Exception as e:
            self.logger.error(f"Error limpiando recursos: {e}")

def setup_logging():
    """Configura logging optimizado"""
    log_format = "%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("bge_m3_multi_gpu.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    # Reducir verbosidad de librerías externas
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

async def main():
    """Función principal asíncrona"""
    setup_logging()
    
    # Verificar GPUs disponibles
    if not torch.cuda.is_available():
        logging.error("CUDA no disponible")
        return
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 4:
        logging.warning(f"Solo se detectaron {gpu_count} GPUs, se esperaban 4")
    
    # Mostrar información del sistema
    logging.info("=" * 60)
    logging.info("SISTEMA DE PROCESAMIENTO BGE-M3 MULTI-GPU")
    logging.info("=" * 60)
    
    for i in range(min(4, gpu_count)):
        gpu_props = torch.cuda.get_device_properties(i)
        logging.info(f"GPU {i}: {gpu_props.name} - {gpu_props.total_memory/1024**3:.1f}GB VRAM")
    
    cpu_count = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024**3
    logging.info(f"CPU: {cpu_count} cores")
    logging.info(f"RAM: {ram_gb:.1f}GB")
    logging.info("=" * 60)
    
    # Crear y ejecutar servicio
    service = OptimizedBgeService()
    await service.process_all_documents()

if __name__ == "__main__":
    # Configurar multiprocessing para CUDA
    mp.set_start_method('spawn', force=True)
    
    # Ejecutar procesamiento asíncrono
    asyncio.run(main())
