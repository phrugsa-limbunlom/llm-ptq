from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class BitsAndBytesInt8Quantizer:
    """
    Quantizer using bitsandbytes library for 8-bit quantization.
    Supports both 8-bit linear layers and 8-bit optimizers.
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.quantized_model = None

    def quantize_model(self, load_in_8bit: bool = True,
                       llm_int8_threshold: float = 6.0,
                       llm_int8_has_fp16_weight: bool = False) -> nn.Module:
        """
        Quantize the model using bitsandbytes 8-bit quantization.
        
        Args:
            load_in_8bit: Whether to load the model in 8-bit precision
            llm_int8_threshold: Threshold for outlier detection in 8-bit quantization
            llm_int8_has_fp16_weight: Whether to keep fp16 weights for outliers
        """
        print("\nQuantize model using bitsandbytes 8-bit quantization...")
        
        # Get the model name from the config
        model_name = self.model.config._name_or_path
        
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            llm_int8_threshold=llm_int8_threshold,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_enable_fp32_cpu_offload = True,
        )
        
        # Create a new model with 8-bit quantization
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # device_map="auto" if torch.cuda.is_available() else None,
        )

        print("8-bit quantization completed successfully!")
        return self.quantized_model

    def save_quantized_model(self, output_path: str):
        """Save the quantized model."""

        print(f"Saving quantized model to {output_path}...")
        self.quantized_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Quantized model saved successfully!") 