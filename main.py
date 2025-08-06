import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from quantization.int8_quantizer import Int8Quantizer

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Post-Training Quantization')
    parser.add_argument('--model_name', type=str, help='Name of the model to quantize')
    
    args = parser.parse_args()
    model_name = args.model_name

    cache_dir = "./output/cache"

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if model files exist in cache
    model_cache_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    model_exists_in_cache = os.path.exists(model_cache_path) and len(os.listdir(model_cache_path)) > 0

    print("Downloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")

    local_files_only = True if model_exists_in_cache else False

    print("Loading model from cache..." if model_exists_in_cache else "Model not found in cache. Downloading model...")

    # Use device_map="auto" if having CUDA enabled and enough VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        # device_map="auto" if torch.cuda.is_available() else None,
    )


    print("Model and tokenizer downloaded successfully!")
    print(f"Model location: {model.config._name_or_path}")

    int8Quant = Int8Quantizer(model, tokenizer)

    calibration_data = int8Quant.prepare_calibration_data()

    quantized_model = int8Quant.quantize_model(calibration_data)

    print("Quantization completed successfully!")
    print(f"Quantized model saved with calibration-based quantization mode selection")