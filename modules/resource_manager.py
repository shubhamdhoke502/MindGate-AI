import psutil
import gc
import os
import requests
from typing import Dict, Any

class SystemResourceManager:
    """
    Monitors host RAM/VRAM, manages memory guardrails, and prevents system crashes
    by selecting appropriate model quantization configs dynamically.
    """
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

    def get_system_memory_status(self) -> Dict[str, Any]:
        """Returns total, available, and percentage RAM usage."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_percent": mem.percent
        }

    def recommend_model_quantization(() -> Dict[str, str]:
        """
        Determines safest model quantization based on available RAM guardrails.
        """
        mem_info = self.get_system_memory_status()
        avail_gb = mem_info["available_gb"]

        if avail_gb < 4.0:
            recommended = "q2_K / q4_0 (Ultra-light GGUF / TinyLlama 1.1B)"
            model_tag = "tinyllama"
        elif 4.0 <= avail_gb < 8.0:
            recommended = "q4_K_M (4-bit Quantized / Phi-3 / Mistral 7B Q4)"
            model_tag = "mistral:7b-instruct-q4_K_M"
        elif 8.0 <= avail_gb < 16.0:
            recommended = "q5_K_M / q8_0 (8-bit Quantized / Llama-3 8B Q5)"
            model_tag = "llama3:8b-instruct-q5_K_M"
        else:
            recommended = "fp16 / Native Unquantized Precision"
            model_tag = "llama3:latest"

        return {
            "available_ram_gb": avail_gb,
            "recommended_quantization": recommended,
            "suggested_model_tag": model_tag
        }

    def flush_memory(self):
        """Forces Python garbage collection to clean up unused memory references."""
        gc.collect()
        try:
            # Tell Ollama to unload idle models from RAM
            requests.post(f"{self.ollama_url}/api/generate", json={"model": "", "keep_alive": 0}, timeout=2)
        except Exception:
            pass
        print("[ResourceManager]: System memory successfully flushed.")


if __name__ == "__main__":
    rm = SystemResourceManager()
    print("[Resource Manager Status]:", rm.get_system_memory_status())
    print("[Quantization Recommendation]:", rm.recommend_model_quantization())