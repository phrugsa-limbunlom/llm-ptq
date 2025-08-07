# LLM Post-Training Quantization

INT8 post-training quantization for Large Language Models with two quantization approaches:
1. **Custom INT8 Quantization**: Calibration-based quantization mode selection
2. **BitsAndBytes INT8 Quantization**: Using the bitsandbytes library

## Features

### Custom INT8 Quantization
- **Weight Quantization**: Symmetric quantization (signed INT8)
- **Activation Quantization**: Calibration-based selection between symmetric/asymmetric modes
- **Automatic Mode Selection**: Uses symmetry score from activation distributions
- **Layer-wise Analysis**: Detailed statistics for each layer during quantization

### BitsAndBytes INT8 Quantization
- **Library-based Quantization**: Uses the bitsandbytes library for optimized 8-bit quantization
- **Outlier Detection**: Configurable threshold for outlier detection
- **FP16 Weight Support**: Option to keep FP16 weights for outliers
- **CPU Offloading**: Automatic FP32 CPU offloading support
- **Memory Optimization**: Efficient memory usage with low CPU memory usage

## How It Works

### Custom INT8 Quantization Process

1. **Data Preparation**: Uses predefined calibration texts to generate representative inputs
2. **Statistics Collection**: Runs forward passes and collects activation statistics for each layer
3. **Symmetry Analysis**: Calculates symmetry score: `symmetry_score = exp(-|mean| / (std + ε))`
4. **Mode Decision**: 
   - `symmetry_score > 0.6` → **Symmetric** quantization
   - `symmetry_score ≤ 0.6` → **Asymmetric** quantization

### Quantization Modes

- **Symmetric**: Signed INT8 (-127 to 127), zero point = 0
- **Asymmetric**: Unsigned INT8 (0 to 255), calculated zero point

### BitsAndBytes Quantization Process

1. **Model Loading**: Loads the model with 8-bit quantization configuration
2. **Outlier Detection**: Identifies and handles outliers based on threshold
3. **Memory Optimization**: Efficient memory management with CPU offloading
4. **Quantization**: Applies 8-bit quantization to linear layers

## Usage

### Custom INT8 Quantization

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

### BitsAndBytes INT8 Quantization

```python
from quantization.bnb_int8_quantizer import BitsAndBytesInt8Quantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Initialize bitsandbytes quantizer
bnb_quantizer = BitsAndBytesInt8Quantizer(model, tokenizer)

# Quantize with custom parameters
quantized_model = bnb_quantizer.quantize_model(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
```

## Run Examples

### Custom INT8 Quantization
```bash
python main.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

### BitsAndBytes INT8 Quantization
```bash
python main.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --use_bnb True
```

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.55.0
- Hugging Face Hub >= 0.16.0
- **BitsAndBytes >= 0.46.1** (for BitsAndBytes quantization)
- **Accelerate >= 1.9.0** (for BitsAndBytes quantization)
- hf_transfer == 0.1.9
- hf-xet == 1.1.5

### Note: Install PyTorch with CUDA
- Check CUDA version → `nvidia-smi`
- For CUDA version 12.4 → `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

## Example Output

### Custom INT8 Quantization Output

```
Downloading tokenizer...
Set padding token to EOS token
Loading model from cache...

Model and tokenizer downloaded successfully!
Model location: meta-llama/Llama-3.1-8B-Instruct

Custom 8-bit Quantization...
Preparing 100 calibration samples...
Running calibration to collect data distribution of each layer...
Layer model.layers.0.self_attn.q_proj: mean=-0.0257, std=1.1416, symmetry_score=0.9777
Layer model.layers.5.mlp.gate_proj: mean=-0.1851, std=0.2786, symmetry_score=0.5146

Layer model.layers.0.self_attn.q_proj: Using SYMMETRIC quantization (symmetry_score: 0.9777)
Original weights - min: -0.746094, max: 0.664062, mean: 0.000000, std: 0.018677
Symmetric quantization - scale: 0.005875
Quantized weights - min: -127, max: 113

Layer model.layers.5.mlp.gate_proj: Using ASYMMETRIC quantization (symmetry_score: 0.5146)
Original weights - min: -0.417969, max: 0.308594, mean: 0.000006, std: 0.014648
Asymmetric quantization - scale: 0.002850, zero_point: 147.0
Quantized weights - min: 0, max: 255

=== Model Size Information ===
original_size_gb: 14.96
quantized_size_gb: 9.01
compression_ratio: 1.66
memory_saved_gb: 5.95
```

### BitsAndBytes INT8 Quantization Output

```
Downloading tokenizer...
Set padding token to EOS token
Loading model from cache...

Model and tokenizer downloaded successfully!
Model location: meta-llama/Llama-3.1-8B-Instruct

8-bit Quantization with BitsAndBytes...
Quantize model using bitsandbytes 8-bit quantization...
8-bit quantization completed successfully!

=== Model Size Information ===
original_size_gb: 14.96
quantized_size_gb: 8.46
compression_ratio: 1.77
memory_saved_gb: 6.50
```

## Configuration Options

### BitsAndBytes Parameters

- `load_in_8bit`: Enable 8-bit quantization (default: True)
- `llm_int8_threshold`: Threshold for outlier detection (default: 6.0)
- `llm_int8_has_fp16_weight`: Keep FP16 weights for outliers (default: False)
- `llm_int8_enable_fp32_cpu_offload`: Enable FP32 CPU offloading (default: True)

### Custom Quantization Parameters

- `symmetry_threshold`: Threshold for symmetric/asymmetric mode selection (default: 0.6)
- `calibration_samples`: Number of calibration samples (default: 100)
- `epsilon`: Small value to prevent division by zero (default: 1e-8) 