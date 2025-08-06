# LLM Post-Training Quantization

INT8 post-training quantization for Large Language Models with calibration-based quantization mode selection.

## Features

- **Weight Quantization**: Symmetric quantization (signed INT8)
- **Activation Quantization**: Calibration-based selection between symmetric/asymmetric modes
- **Automatic Mode Selection**: Uses symmetry score from activation distributions

## How It Works

### Calibration Process

1. **Data Preparation**: Uses predefined calibration texts to generate representative inputs
2. **Statistics Collection**: Runs forward passes and collects activation statistics for each layer
3. **Symmetry Analysis**: Calculates symmetry score: `symmetry_score = exp(-|mean| / (std + ε))`
4. **Mode Decision**: 
   - `symmetry_score > 0.6` → **Symmetric** quantization
   - `symmetry_score ≤ 0.6` → **Asymmetric** quantization

### Quantization Modes

- **Symmetric**: Signed INT8 (-128 to 127), zero point = 0
- **Asymmetric**: Unsigned INT8 (0 to 255), calculated zero point

## Usage

```python
from quantization.int8_quantizer import Int8Quantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False
    )

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

# Initialize quantizer
quantizer = Int8Quantizer(model, tokenizer)

# Prepare calibration data and quantize
calibration_data = quantizer.prepare_calibration_data()
quantized_model = quantizer.quantize_model(calibration_data)
```

## Run Example

```bash
python main.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Hugging Face Hub >= 0.16.0

#### Note: Install pytorch with CUDA
- Check CUDA version -> ```nvidia-smi```
- For CUDA version 12.4 -> ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```

## Example Output

```
Downloading tokenizer...
Set padding token to EOS token
Loading model from cache...

Model and tokenizer downloaded successfully!
Model location: meta-llama/Llama-3.1-8B-Instruct

Preparing 100 calibration samples...
Running calibration to collect data distribution of each layer...
Layer model.layers.0.self_attn.q_proj: mean=-0.0257, std=1.1416, symmetry_score=0.9777
Layer model.layers.4.mlp.gate_proj: mean=-0.1658, std=0.2615, symmetry_score=0.5305

Lyer model.layers.0.self_attn.q_proj: Using SYMMETRIC quantization (symmetry_score: 0.9777)
Original weights - min: -0.746094, max: 0.664062, mean: 0.000000, std: 0.018677
Symmetric quantization - scale: 0.002926
Quantized weights - min: -128, max: 127

Layer model.layers.4.mlp.gate_proj: Using ASYMMETRIC quantization (symmetry_score: 0.5305)
Original weights - min: -0.316406, max: 0.392578, mean: -0.000014, std: 0.014679
Asymmetric quantization - scale: 0.001539, zero_point: 0.0
Quantized weights - min: 0, max: 255
``` 