# 🌟 OpenMisty

OpenMisty is a quantized and optimized version of [OpenManus](https://github.com/mannaandpoem/OpenManus), designed to run AGI locally on any computer, even with limited resources.

## 🚀 Features

- **Quantized Models**: Run powerful LLMs with reduced memory footprint
- **Local Execution**: No need for cloud resources or API keys (optional)
- **Multi-Model Support**: Use various quantized models based on your hardware
- **Optimized Performance**: Faster inference with minimal quality loss
- **Full AGI Capabilities**: Retain all the capabilities of OpenManus
- **Hardware-Aware**: Automatically adapts to your hardware capabilities

## 🔧 Installation

### Method 1: Using pip

```bash
pip install openmisty
```

### Method 2: From source

```bash
git clone https://github.com/iamkrishnagupta10/OpenMistyQuantized.git
cd OpenMistyQuantized
pip install -e .
```

### Method 3: Using conda

```bash
conda create -n openmisty python=3.12
conda activate openmisty
git clone https://github.com/iamkrishnagupta10/OpenMistyQuantized.git
cd OpenMistyQuantized
pip install -e .
```

## ⚙️ Configuration

OpenMisty can be configured to use either local models or API-based models. Create a `config.toml` file in the `config` directory:

```bash
cp config/config.example.toml config/config.toml
```

Edit the configuration file to choose your preferred model and settings:

```toml
# Global LLM configuration
[llm]
# Choose between "local" or "api" mode
mode = "local"
# For local mode, specify the model path or HuggingFace model ID
model = "mistralai/Mistral-7B-Instruct-v0.2"
# Quantization level: 4bit, 8bit, or none
quantization = "4bit"
# For API mode, specify the API details
api_key = ""
base_url = "https://api.openai.com/v1"
max_tokens = 4096
temperature = 0.0
```

## 🏃‍♂️ Quick Start

Run OpenMisty with a single command:

```bash
openmisty
```

Or from the source directory:

```bash
python main.py
```

## 🧠 Supported Models

OpenMisty supports various models that can be quantized:

- Mistral (7B)
- Llama 3 (8B)
- Phi-3 (3.8B, 7B)
- Gemma (2B, 7B)
- Falcon (7B)
- And more!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [OpenManus](https://github.com/mannaandpoem/OpenManus) for the original implementation
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization support 