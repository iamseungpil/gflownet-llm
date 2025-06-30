import torch
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="sft_data_74dd1130.jsonl",
        metadata={"help": "Path to the training data"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    prompt_template: str = field(
        default="llama3",
        metadata={"help": "Prompt template to use: 'llama3' or 'custom'"}
    )

@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(default="./o2arc-llama3-sft")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    optim: str = field(default="paged_adamw_8bit")
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    report_to: str = field(default="wandb")
    run_name: str = field(default="o2arc-llama3-sft")

def format_prompt_llama3(example: Dict) -> str:
    """Format prompt for Llama-3.1 instruction format"""
    system_prompt = """You are an AI assistant specialized in analyzing grid transformations. 
Given an initial grid and a final grid, you need to identify the exact sequence of actions that transforms the initial grid into the final grid.

Available actions:
- Flip(vertical/horizontal): Flips the grid along the specified axis
- Rotate(clockwise/counterclockwise): Rotates the grid 90 degrees
- Paint(color=X): Changes the color of selected cell(s)
- SelectCell(x=X, y=Y): Selects a specific cell
- SelectGrid: Selects the entire grid

Respond with only the action sequence, separated by ' -> '."""
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    
    return prompt

def format_prompt_custom(example: Dict) -> str:
    """Custom prompt format for grid transformation"""
    prompt = f"""### Task: Identify Grid Transformation Actions

{example['input']}

### Answer: {example['output']}"""
    
    return prompt

def load_dataset(data_path: str, prompt_template: str = "llama3") -> Dataset:
    """Load and format dataset for training"""
    data = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            
            # Format based on template
            if prompt_template == "llama3":
                formatted_text = format_prompt_llama3(example)
            else:
                formatted_text = format_prompt_custom(example)
            
            data.append({
                "text": formatted_text,
                "input": example["input"],
                "output": example["output"]
            })
    
    # Split into train and eval (90/10 split)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    print(f"Loaded {len(train_data)} training examples and {len(eval_data)} evaluation examples")
    
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)

def get_model_and_tokenizer(model_args: ModelArguments):
    """Load model and tokenizer with optional quantization"""
    
    # Quantization config
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA if enabled
    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsCustom))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize wandb
    if training_args.report_to == "wandb":
        wandb.init(project="o2arc-sft", name=training_args.run_name)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(model_args)
    
    # Load datasets
    print(f"Loading dataset from {data_args.data_path}...")
    train_dataset, eval_dataset = load_dataset(data_args.data_path, data_args.prompt_template)
    
    # Data collator for completion only training
    # This ensures we only compute loss on the assistant's response
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=data_args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluate on test examples
    print("\n=== Evaluation Examples ===")
    model.eval()
    
    for i, example in enumerate(eval_dataset.select(range(min(3, len(eval_dataset))))):
        # Prepare input
        input_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant specialized in analyzing grid transformations. 
Given an initial grid and a final grid, you need to identify the exact sequence of actions that transforms the initial grid into the final grid.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = generated.split("assistant")[-1].strip()
        
        print(f"\nExample {i+1}:")
        print(f"Expected: {example['output']}")
        print(f"Predicted: {prediction}")
        print("-" * 50)

if __name__ == "__main__":
    main()
