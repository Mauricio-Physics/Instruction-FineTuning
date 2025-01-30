import argparse
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer, 
        TrainingArguments,
        BitsAndBytesConfig, 
        Trainer, 
        AutoConfig)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
import json
from trl.trainer import ConstantLengthDataset
import torch


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    #Model Input parameters
    parser.add_argument('--model_id', type=str, default="mistralai/Mistral-7B-v0.3", help='Model\'s name on HuggingFace')
    parser.add_argument('--quantization', type=str, default='4', help='Number of quantization bits (16,8,4)')
    parser.add_argument('--cache_dir', type=str, default='/root/GenAI/.huggingface', help='Directory in which are saved \
                        the models and the dataset downloaded from HF')
    
    #PEFT parameters (Lora)
    parser.add_argument('--lora_rank', type=int, default=32, help='Lora Rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=8, help='Lora alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=.3, help='Lora Dropout parameter')

    #Training parameters
    parser.add_argument('--lr', type=float, default=10**-4, help='Learning rate of the training algorithm')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size used in the training algorithm')
    parser.add_argument('--gradient_accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epoch')
    parser.add_argument('--do_eval', type=str, default='False', help='Perform evaluation at the end of each epoch')

    #Output directory
    parser.add_argument('--output_dir', type=str, help='Directory where the model will be saved')
    parser.add_argument('--push_to_hub', type=str, default='False', help='HuggingFace repository where the model will be saved')
    parser.add_argument('--last_checkpoint', type=str, default='', help='Resume from last checkpoint')
    return parser.parse_args()


def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)


def prompt_generation(row):
    return ("[INST] Esegui il task richiesto nella sezione \"Istruzione\"."
            "### Istruzione:\n {system}\n\n ### Domadnda:\n{question} ### Contesto:\n{context}\n\n{chunks_rag}[\INST] ###Risposta:\n{answer}").format_map(row)


def create_prompt(row, tokenizer=None):
    if tokenizer is None:
        return False
    prompt = prompt_generation(row)
    endcoding_dict = tokenizer(prompt, max_length=8000, padding='max_length', truncation=True)
    return {'input_ids':endcoding_dict['input_ids'],
            'attention_mask': endcoding_dict['attention_mask'],
            'labels': endcoding_dict['input_ids'].copy()}


def dataset_processing(tokenizer, data_path='data/answers/third_batch/final/'):
    data = load_from_disk(data_path)
    data_dict = data.train_test_split(test_size=0.1)
    train_data = data_dict['train']
    eval_data = data_dict['test']
    train_dataset = train_data.map(create_prompt, fn_kwargs={"tokenizer": tokenizer})
    valid_dataset = eval_data.map(create_prompt, fn_kwargs={"tokenizer": tokenizer})
    train_dataset = train_dataset.remove_columns(['system', 'question', 'context', 'answer', 'chunks_rag'])
    valid_dataset = valid_dataset.remove_columns(['system', 'question', 'context', 'answer', 'chunks_rag'])
    return train_dataset, valid_dataset


def model_loader(model_id, quantization, lora_rank, lora_alpha, lora_dropout, cache_dir):
    '''
    Load the model according to the selection criteria made in command line
    '''

    config_kwargs = {
    "cache_dir": cache_dir,
    "revision": 'main'
    }
    config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    bnb_config = None
    if quantization=="4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    if quantization=="8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    model = None
    if bnb_config is not None:
        model =  AutoModelForCausalLM.from_pretrained(model_id, config = config, quantization_config=bnb_config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model =  AutoModelForCausalLM.from_pretrained(model_id, config = config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    model = get_peft_model(model, peft_config)
    tokenizer_kwargs = {
    "cache_dir": cache_dir,
    "use_fast": True,
    "revision": 'main',
    "max_length": 8000,
    "padding_side":"left",
    "add_eos_token": True,
    'truncation': True
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    DEFAULT_CHAT_TEMPLATE = "[INST] {}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, valid_dataset, lr, num_epochs, gradient_accumulation, batch_size, output_dir, do_eval,
                last_checkpoint):
    last_checkpoint = None if last_checkpoint == '' else last_checkpoint
    training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio = 0.1,
            gradient_accumulation_steps=gradient_accumulation,
            gradient_checkpointing=True,
            num_train_epochs=num_epochs,
            logging_strategy="steps",
            logging_steps=1,
            save_steps=10,
            save_strategy="steps",
            evaluation_strategy='steps',
            do_eval= do_eval,
            eval_steps = 230,
            optim = "adamw_torch",
            )
    tokenizer.padding_side = 'right'
    trainer = Trainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset)
    print("Training...")
    if last_checkpoint is not None:
        trainer.train(output_dir+last_checkpoint)
    else:
        trainer.train()

def main():
    torch.manual_seed(22)
    args = parse_command_line_arguments()
    model, tokenizer = model_loader(model_id = args.model_id, quantization=args.quantization, lora_rank=args.lora_rank, 
                        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, cache_dir=args.cache_dir)
    train_dataset, valid_dataset = dataset_processing(tokenizer=tokenizer)
    train_model(model=model, tokenizer=tokenizer, train_dataset=train_dataset, valid_dataset=valid_dataset, num_epochs=args.num_epochs,
                lr=args.lr, gradient_accumulation=args.gradient_accumulation, batch_size=args.batch_size, output_dir='./finetuned_model/', do_eval=args.do_eval,
                last_checkpoint=args.last_checkpoint)


if __name__ == "__main__":
    main()
