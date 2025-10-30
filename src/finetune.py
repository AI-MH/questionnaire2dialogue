from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
import argparse
import torch

def main(dataset_path, out_model_name, epochs=5):
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def create_instruction(sample):
        prompt = f"{sample['input']}"
        return {"text" : f"{prompt} {sample['output']}"}

    print(dataset_path)
    try:
        dataset = load_from_disk(dataset_path)["train"]
    except ValueError:
        dataset = load_dataset(dataset_path, split="train")

    dataset = dataset.map(create_instruction, remove_columns=dataset.features, batched=False)

    tokenizer.pad_token = tokenizer.eos_token

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer.padding_side = 'right' # to prevent warnings

    args = TrainingArguments(
        seed=666,
        output_dir=out_model_name, # directory to save and repository id
        num_train_epochs=epochs,                     # number of training epochs
        per_device_train_batch_size=2,          # batch size per device during training
        gradient_accumulation_steps=16,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-5,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model
    trainer.save_model()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-p", "--path", type=str, required=True, help="The name/path of the data to train on.")
    args.add_argument("-o", "--out_model_name", type=str, required=True, help="The name/path of the model to evaluate.")
    args.add_argument("-e", "--epochs", type=int, required=False, default=5, help="The name/path of the model to evaluate.")
    args = args.parse_args()
    main(args.path, args.out_model_name, args.epochs)