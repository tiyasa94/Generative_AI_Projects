# Fine-tuning DeepSeek Model for Finance Domain

This guide walks you through the steps to fine-tune the **DeepSeek model** for finance-specific tasks using **unsloth** and other required dependencies.

## 1. Environment Setup

### Install NVIDIA Driver and Dependencies

To verify your GPU installation and ensure it’s ready for model fine-tuning, use the following command:

```bash
nvidia-smi
```
### Install Required Python Libraries

To install the required Python packages for this project, you need to install the following:

```bash
pip3 install torch
pip3 install numpy
pip install unsloth
pip uninstall unsloth -y
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## 2. Load Pre-trained DeepSeek Model

### Initialize the Model

Once the dependencies are installed, load the pre-trained DeepSeek model using unsloth. The model is loaded with a maximum sequence length of 2048 and configured to load in 4-bit precision for optimized memory usage.

*  model_name: Specifies the pre-trained model to use.
*  max_seq_length: Defines the maximum token length for each input.
*  load_in_4bit: Optimizes memory usage by loading the model in 4-bit precision.

## 3. Apply PEFT (Parameter-Efficient Fine-Tuning) to the Model

### Apply PEFT Configuration

To make the fine-tuning process more efficient, we apply **PEFT** (Parameter-Efficient Fine-Tuning) to the pre-trained **DeepSeek** model. This step involves configuring the model with specific settings that allow efficient training while keeping memory usage minimal.

We apply PEFT to the pre-trained model by configuring the parameters like `lora_alpha`, `lora_dropout`, and others. Here's how to do it:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # Rank of the low-rank adaptation matrices
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target model modules for adaptation
    lora_alpha=16,  # Alpha for scaling in LoRA (Low-Rank Adaptation)
    lora_dropout=0,  # Dropout rate for LoRA
    bias="none",  # Disable bias for parameter-efficient training
    use_gradient_checkpointing="unsloth",  # Enables gradient checkpointing for memory efficiency
    random_state=42,  # Random seed for reproducibility
    use_rslora=False,  # Disables the use of RSLoRA (Revised LoRA) if False
    loftq_config=None,  # Optional configuration for LoFTQ
)
```

## 4. Load the Finance Dataset 

To fine-tune the model for the finance domain, we need a dataset that contains finance-related content. In this step, we load the **Finance-Instruct-500k** dataset, which is specifically designed for finance tasks.

We use the `datasets` library to load the dataset. This library provides access to many pre-built datasets for NLP tasks.

```python
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
```


