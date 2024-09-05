import os
import torch
from unsloth import (
    FastLanguageModel,
    to_sharegpt,
    standardize_sharegpt,
    apply_chat_template,
)
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
import fire


class FinetuneModel:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        prompt_format_path: str,
        chat_format_path: str,
        batch_size: int = 1,
        hf_repo: str = "your_name/lora_model",
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prompt_format_path = prompt_format_path
        self.chat_format_path = chat_format_path
        self.batch_size = batch_size
        self.hf_repo = hf_repo

        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True

        self.load_model()
        self.load_dataset()
        self.train_model()
        self.evaluate()
        self.save_model()

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)
        with open(self.prompt_format_path, "r") as file:
            merge_prompt = file.read()

        with open(self.chat_format_path, "r") as file:
            chat_template = file.read()

        for key in ["train", "validation", "test"]:
            self.dataset[key] = to_sharegpt(
                self.dataset[key],
                merged_prompt=merge_prompt,
                output_column_name="Answer",
            )

            self.dataset[key] = standardize_sharegpt(self.dataset[key])

            self.dataset["train"] = apply_chat_template(
                self.dataset["train"],
                tokenizer=self.tokenizer,
                chat_template=chat_template,
            )

    def train_model(self):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=5,
                num_train_epochs=4,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="wandb",
                output_dir="outputs",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                eval_strategy="steps",
                save_steps=50,
                save_total_limit=2,  # save only current and best
                eval_steps=10,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

    def evaluate(self):
        # import wandb
        #
        # results = self.model.evaluate(
        #     eval_dataset=self.dataset["test"]
        # )  # Evaluate on test split
        # print(results)
        # wandb.log(results)
        pass

    def save_model(self):
        self.model.save_pretrained("lora_model")
        self.tokenizer.save_pretrained("lora_model")
        self.model.push_to_hub_gguf(
            self.hf_repo,
            self.tokenizer,
            quantization_method=[
                "q4_k_m",
                # "q8_0",
                # "q5_k_m",
            ],
        )


if __name__ == "__main__":
    fire.Fire(FinetuneModel)
