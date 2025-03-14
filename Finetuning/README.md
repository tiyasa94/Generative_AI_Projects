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

## 5. Convert Dataset to ShareGPT Format

To make the dataset suitable for fine-tuning, we need to preprocess it into a conversation-like format. This step involves using the **`to_sharegpt`** function from **unsloth** to format the dataset into a structure that aligns with the conversation flow.

We use the **`to_sharegpt`** function to convert the dataset into the correct format for training, where the input and output are aligned in a conversation-style format.

```python
dataset = to_sharegpt(
    dataset,
    merged_prompt="{user}[[\nYour input is:\n{system}]]",  # Merged prompt using available columns
    output_column_name="assistant",  # Output column name to be used for the assistant's response
    conversation_extension=3,  # Extend the conversation by 3 steps
)
```

## 6. Standardize the Dataset for ShareGPT

After converting the dataset into the ShareGPT format, it’s important to standardize the structure of the dataset to ensure consistency. The **`standardize_sharegpt`** function from **unsloth** is used to achieve this.

We use the **`standardize_sharegpt`** function to ensure that the dataset follows a consistent structure for each conversation. This function will ensure that all conversations are formatted correctly before fine-tuning the model.

```python
dataset = standardize_sharegpt(dataset)
```

## 7. Fine-Tune the Model Using SFTTrainer

After preparing and standardizing the dataset, the next step is fine-tuning the model using the **SFTTrainer** class from **TRL** (Transformers Reinforcement Learning). This class simplifies the training process by handling much of the underlying complexity.

### SFTTrainer Configuration

The **SFTTrainer** is designed to manage the fine-tuning process efficiently. It accepts several parameters that control how the model will be trained, including the dataset, batch sizes, learning rate, and the model itself.

Key parameters for **SFTTrainer** include:

- **Model and Tokenizer**: You will provide the pre-trained model and its associated tokenizer. These will be used to process the input data and generate the model’s output.
  
- **Training Dataset**: The dataset you’ve prepared and standardized is passed to the trainer, ensuring that the model receives structured input that matches its training requirements.

- **TrainingArguments**: This is a collection of hyperparameters that control how the training will proceed. It includes settings like:
  - **Batch Size**: Determines how many samples the model processes before updating its weights. This is an important factor that can affect both training speed and memory usage.
  - **Learning Rate**: The rate at which the model adjusts its weights. A high learning rate can lead to instability, while a low rate might make the training process slower.
  - **Number of Training Steps**: Specifies how many steps the model will take during training. More steps usually result in better fine-tuning but require more computational resources.
  - **Gradient Accumulation**: This allows the model to accumulate gradients over multiple steps before performing a backward pass, which helps when working with larger models or smaller batch sizes.
  - **Precision**: Mixed-precision training (16-bit or bfloat16) can be enabled for more memory-efficient training. This is particularly helpful for training large models on hardware with limited memory.

- **Other Optimizations**: 
  - **Gradient Checkpointing**: This technique allows you to reduce memory usage during backpropagation by saving intermediate results only when necessary, allowing you to train larger models.
  - **Weight Decay**: This helps prevent overfitting by penalizing large weights during training, ensuring that the model generalizes well to unseen data.


## 8. Start Training the Model

Once the **SFTTrainer** is configured, the next step is to begin the training process. This is done by calling the **`train()`** method on the **SFTTrainer** instance. This will initiate the fine-tuning of the model on your dataset using the parameters you have specified.

### Training the Model

By calling the `train()` method, the model begins to learn from the dataset by adjusting its parameters (weights) to minimize the loss function over the course of several training steps.

```python
trainer_stats = trainer.train()
```

## 9. Install Ollama

To integrate **Ollama** into your project, you need to install it on your system. Ollama is a platform that enables running large language models locally, which can be useful for fine-tuning and inference tasks.

You can install Ollama using the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 10. Save the Fine-Tuned Model

After training the model, it’s essential to save the fine-tuned model for later use or deployment. The **`save_pretrained_gguf()`** method allows you to save the model in the **GGUF** (Generic GPT Unified Format), which is a format optimized for easier loading and sharing of large language models.

You can save the fine-tuned model using the following command:

```python
model.save_pretrained_gguf("model", tokenizer)
```

## 11. Start the Ollama Server

After saving your fine-tuned model, you may want to serve it locally for inference or integration into applications. You can start the Ollama server, which will run the model and handle incoming requests.

### Start Ollama Server

You can start the Ollama server using the following Python code:

```python
subprocess.Popen(["ollama", "serve"])
time.sleep(3)
```

## 12. Create the Model in Ollama

After fine-tuning your model and saving it in the GGUF format, you can create a model in Ollama using the saved model file. This step makes the fine-tuned model available for inference and further use.

You can create the model in Ollama with the following command:

```bash
ollama create deepseek_finetuned_finance_model_TM -f ./model/Modelfile
```

## 13. Interact with the Fine-Tuned Model

Once the model is created in Ollama, you can interact with it to perform inference tasks. The **Ollama** library provides a simple interface to send chat-style queries to the model.

### Send a Query to the Fine-Tuned Model

To interact with the fine-tuned model, use the following Python code:

```python
response = ollama.chat(
    model="deepseek_finetuned_finance_model_TM",  # Name of the fine-tuned model
    messages=[{
        "role": "user",
        "content": "What is the difference between home loan and consumer loan? Explain with example."
    }]
)
print(response.message.content)
```

## Conclusion

In this guide, we've walked through the entire process of fine-tuning the **DeepSeek** model for the **finance domain**, from setting up the environment and loading the dataset to fine-tuning the model and deploying it for inference. You have learned how to:

- Install necessary dependencies and tools such as **Ollama**.
- Load and preprocess a finance-specific dataset for training.
- Fine-tune the model using **unsloth** and the **SFTTrainer**.
- Standardize and prepare the dataset for serving with **ShareGPT**.
- Save and create the fine-tuned model in Ollama.
- Interact with the model by sending chat-style queries.

By following these steps, you now have a **fine-tuned DeepSeek model** tailored for the finance domain, ready to be deployed for various real-time applications such as answering finance-related questions, generating financial insights, and more.

The knowledge gained from this guide can be used to further enhance the model, integrate it into larger systems, or apply it to other specialized domains. With the foundation laid, you can confidently move forward with deploying and scaling your finance-specific AI model.





