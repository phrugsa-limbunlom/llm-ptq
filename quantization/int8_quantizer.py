from enum import Enum
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn


class QuantizationMode(Enum):
    """Enumeration for quantization modes."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class CustomInt8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, quantization_mode: QuantizationMode):

        super(CustomInt8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.quantization_mode = quantization_mode
        self.symmetric_qmin, self.symmetric_qmax = -2 ** (8 - 1), 2 ** (8 - 1) - 1  # -128, 127
        self.asymmetric_qmin, self.asymmetric_qmax = 0, (2 ** 8) - 1  # 0, 255

        self.register_buffer('qweight', torch.zeros(
            (out_features, in_features), dtype=torch.int8
        ))

    def quantize_weights(self, weight: torch.Tensor):
        """Quantization using the determined mode (symmetric or asymmetric)."""

        original_min = torch.min(weight).item()
        original_max = torch.max(weight).item()
        original_mean = torch.mean(weight).item()
        original_std = torch.std(weight).item()

        print(
            f"Original weights - min: {original_min:.6f}, max: {original_max:.6f}, mean: {original_mean:.6f}, std: {original_std:.6f}")

        if self.quantization_mode == QuantizationMode.SYMMETRIC:
            quantized_weights = self._quantize_symmetric(weight)
            print(f"Symmetric quantization - scale: {self.weight_scale:.6f}")
        else:
            quantized_weights = self._quantize_asymmetric(weight)
            print(
                f"Asymmetric quantization - scale: {self.activation_scale:.6f}, zero_point: {self.activation_zero_point}")

        # Print quantized weight statistics
        quantized_min = torch.min(quantized_weights).item()
        quantized_max = torch.max(quantized_weights).item()

        print(f"Quantized weights - min: {quantized_min}, max: {quantized_max}")

        return quantized_weights

    def _quantize_symmetric(self, real_value: torch.Tensor):
        """Symmetric quantization for weights with signed INT8."""
        real_max = torch.max(torch.abs(real_value))
        real_min = torch.min(torch.abs(real_value))

        scale = (real_max - real_min) / (self.symmetric_qmax - self.symmetric_qmin)

        q = torch.round(real_value / scale).clamp(self.symmetric_qmin, self.symmetric_qmax).to(torch.int8)

        self.register_buffer('qweight', q)
        self.weight_scale = scale

        return q

    def _quantize_asymmetric(self, real_value: torch.Tensor):
        """Asymmetric quantization for activations."""
        real_max = torch.max(torch.abs(real_value))
        real_min = torch.min(torch.abs(real_value))

        scale = (real_max - real_min) / (self.asymmetric_qmax - self.asymmetric_qmin)
        zero_point = self.asymmetric_qmin - torch.round(real_min / scale)

        # Clamp zero point to valid range
        zero_point = torch.clamp(zero_point, self.asymmetric_qmin, self.asymmetric_qmax)

        q = (torch.round(real_value / scale) + zero_point).clamp(
            self.asymmetric_qmin, self.asymmetric_qmax
        ).to(torch.uint8)

        self.register_buffer('qweight', q)

        self.activation_scale = scale
        self.activation_zero_point = zero_point

        return q


class Int8Quantizer:
    def __init__(self, model: nn.Module, tokenizer):
        self.quantized_model = None
        self.tokenizer = None
        self.layer_distribution = {}  # Store data distribution for each layer

        self.model = model
        self.tokenizer = tokenizer

    def prepare_calibration_data(self, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Prepare calibration data for quantization."""
        print(f"Preparing {num_samples} calibration samples...")

        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Quantization reduces model size while preserving performance.",
            "Transformers have revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "Post-training quantization is applied after model training.",
            "Model compression techniques help deploy large models efficiently."
        ]

        all_texts = []
        for _ in range((num_samples // len(calibration_texts)) + 1):
            all_texts.extend(calibration_texts)
        all_texts = all_texts[:num_samples]

        inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        return inputs

    def quantize_model(self, calibration_data: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
        # Run calibration to determine activation layers
        if calibration_data is not None:
            self._run_calibration(calibration_data)

        # Replace full-precision linear layers with INT8 layers
        self.quantized_model = self._replace_linear_layers(self.model)

        return self.quantized_model

    def _collect_layer_distribution(self, module: nn.Module, output_tensor: torch.Tensor, name: str):
        """Collect layer distribution for determining quantization mode."""
        if isinstance(module, nn.Linear):
            # Calculate statistics
            # data = output_tensor.detach().cpu().numpy().flatten()

            mean = torch.mean(output_tensor).item()
            std = torch.std(output_tensor).item()

            # Calculate symmetry score
            normalized_offset = abs(mean) / (std + 1e-8)
            symmetry_score = np.exp(-normalized_offset)  # Maps to (0,1], 1=perfectly symmetric

            # Store statistics
            self.layer_distribution[name] = {
                'mean': mean,
                'std': std,
                'symmetry_score': symmetry_score
            }

            print(f"Layer {name}: mean={mean:.4f}, std={std:.4f}, symmetry_score={symmetry_score:.4f}")

    def _register_hooks(self, module: nn.Module):
        """Register hooks to collect layers distribution."""
        for name, child in module.named_modules():

            if isinstance(child, nn.Linear):
                def create_hook(name):
                    return lambda m, i, o, n=name: self._collect_layer_distribution(m, o, n)

                child.register_forward_hook(create_hook(name))

    def _run_calibration(self, calibration_data: Dict[str, torch.Tensor]):
        """Run calibration to collect layers distribution."""
        print("Running calibration to collect data distribution of each layer...")

        # Register hooks for all linear layers
        self._register_hooks(self.model)

        # Run forward pass with calibration data
        with torch.no_grad():
            self.model(**calibration_data)

        print(f"Collected statistics for {len(self.layer_distribution)} layers")

    def _replace_linear_layers(self, module: nn.Module) -> nn.Module:
        print("Replacing linear layers with quantization...")

        # Collect all modules that need to be replaced
        replacements = []
        for name, child in module.named_modules():
            # Determine quantization mode of each layer based on calibration data
            if name in self.layer_distribution:
                symmetry_score = self.layer_distribution[name]['symmetry_score']
                quantization_mode = QuantizationMode.SYMMETRIC if symmetry_score > 0.6 else QuantizationMode.ASYMMETRIC

                print(f"\nLayer {name}: Using {quantization_mode.name} quantization (symmetry_score: {symmetry_score:.4f})")

                custom_layer = CustomInt8Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    quantization_mode=quantization_mode
                )

                if child.weight is not None:
                    custom_layer.quantize_weights(child.weight.data)

                replacements.append((name, custom_layer))

        # Replace all modules with quantization
        for name, custom_layer in replacements:
            # Find the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                # Get the parent module
                parent = module
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, attr_name, custom_layer)
            else:
                # Root level module
                setattr(module, attr_name, custom_layer)

        return module
