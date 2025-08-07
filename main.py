import os
import argparse
from typing import Dict

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from quantization.bnb_int8_quantizer import BitsAndBytesInt8Quantizer
from quantization.int8_quantizer import Int8Quantizer


def get_model_size_info(model: nn.Module, quantized_model: nn.Module) -> Dict[str, float]:
    """Get information about model size before and after quantization."""

    # Calculate model sizes
    def get_model_size(model):
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size + buffer_size

    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size

    return {
        "original_size_gb": original_size / (1024 * 1024 * 1024),
        "quantized_size_gb": quantized_size / (1024 * 1024 * 1024),
        "compression_ratio": compression_ratio,
        "memory_saved_gb": (original_size - quantized_size) / (1024 * 1024 * 1024),
    }

def save_quantized_model(quantized_model: nn.Module, tokenizer, output_path: str):
    """Save the quantized model."""

    print(f"Saving quantized model to {output_path}...")
    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Quantized model saved successfully!")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Post-Training Quantization with BitsAndBytes')
    parser.add_argument('--model_name', type=str, help='Name of the model to quantize')
    parser.add_argument('--use_bnb', type=bool, default=False, help='Flag to use BityAndBytes quantizer')

    args = parser.parse_args()
    model_name = args.model_name

    cache_dir = "./output/cache"
    output_dir = "./output/quantized_model"

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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

    if not args.use_bnb:

        print("\nCustom 8-bit Quantization...")

        int8Quant = Int8Quantizer(model, tokenizer)

        calibration_data = int8Quant.prepare_calibration_data()

        quantized_model = int8Quant.quantize_model(calibration_data)

        print("Quantization completed successfully!")

    else:
        print("\n8-bit Quantization with BitsAndBytes...")

        # Initialize the bitsandbytes quantizer
        bnb_quantizer = BitsAndBytesInt8Quantizer(model)

        # Quantize the model using bitsandbytes
        quantized_model = bnb_quantizer.quantize_model(
            llm_int8_threshold=0.6,
            llm_int8_has_fp16_weight=False
        )

    # Get model size information
    size_info = get_model_size_info(model=model, quantized_model=quantized_model)
    print("\n=== Model Size Information ===")
    for key, value in size_info.items():
        print(f"{key}: {value:.2f}")

    # Save the quantized model (if having enough space)
    # save_quantized_model(quantized_model, tokenizer, output_dir)
