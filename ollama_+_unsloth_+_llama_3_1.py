# -*- coding: utf-8 -*-
"""Ollama + Unsloth + Llama-3.1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ijuInungAriEVhyLxYIb7EAXkcwfokqi

To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
  <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://ollama.com/"><img src="https://raw.githubusercontent.com/unslothai/unsloth/nightly/images/ollama.png" height="44"></a>
  <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
  <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth#installation-instructions---conda).

You will learn how to do [data prep](#Data) and import a CSV, how to [train](#Train), how to [run the model](#Inference), & [how to export to Ollama!](#Ollama)

[Unsloth](https://github.com/unslothai/unsloth) now allows you to automatically finetune and create a [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md), and export to [Ollama](https://ollama.com/)! This makes finetuning much easier and provides a seamless workflow from `Unsloth` to `Ollama`!

**[NEW]** We now allow uploading CSVs, Excel files - try it [here](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing) by using the Titanic dataset.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#
# # We have to check which Torch version for Xformers (2.3 -> 0.0.27)
# from torch import __version__; from packaging.version import Version as V
# xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
# !pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton
"""* We support Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermes etc
* We support 16bit LoRA or 4bit QLoRA. Both 2x faster.
* `max_seq_length` can be set to anything, since we do automatic RoPE Scaling via [kaiokendev's](https://kaiokendev.github.io/til) method.
* With [PR 26037](https://github.com/huggingface/transformers/pull/26037), we support downloading 4bit models **4x faster**! [Our repo](https://huggingface.co/unsloth) has Llama, Mistral 4bit models.
* [**NEW**] We make Phi-3 Medium / Mini **2x faster**! See our [Phi-3 Medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)
"""

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

from datasets import load_dataset

dataset = load_dataset("britojr/sec10q_v2", split="train")
print(dataset.column_names)

"""One issue is this dataset has multiple columns. For `Ollama` and `llama.cpp` to function like a custom `ChatGPT` Chatbot, we must only have 2 columns - an `instruction` and an `output` column.

To solve this, we shall do the following:
* Merge all columns into 1 instruction prompt.
* Remember LLMs are text predictors, so we can customize the instruction to anything we like!
* Use the `to_sharegpt` function to do this column merging process!

For example below in our [Titanic CSV finetuning notebook](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing), we merged multiple columns in 1 prompt:

<img src="https://raw.githubusercontent.com/unslothai/unsloth/nightly/images/Merge.png" height="100">

To merge multiple columns into 1, use `merged_prompt`.
* Enclose all columns in curly braces `{}`.
* Optional text must be enclused in `[[]]`. For example if the column "Pclass" is empty, the merging function will not show the text and skp this. This is useful for datasets with missing values.
* You can select every column, or a few!
* Select the output or target / prediction column in `output_column_name`. For the Alpaca dataset, this will be `output`.

To make the finetune handle multiple turns (like in ChatGPT), we have to create a "fake" dataset with multiple turns - we use `conversation_extension` to randomnly select some conversations from the dataset, and pack them together into 1 conversation.
"""

merge_prompt = """
Answer questions with short factoid answers.

---

Follow the following format.


Question: question

Answer: uses financial vocabulary

---

Context: {context_1}\n{context_2}\n{context_3}

Question: {Question}

Reasoning: Let's think step by step in order to
"""

from unsloth import to_sharegpt

dataset = to_sharegpt(
    dataset,
    merged_prompt=merge_prompt,
    output_column_name="Answer",
)

"""Finally use `standardize_sharegpt` to fix up the dataset!"""

from unsloth import standardize_sharegpt

dataset = standardize_sharegpt(dataset)

"""### Customizable Chat Templates

You also need to specify a chat template.

You have to use `{INPUT}` for the instruction and `{OUTPUT}` for the response.

We also allow you to use an optional `{SYSTEM}` field. This is useful for Ollama when you want to use a custom system prompt (also like in ChatGPT).

You can also not put a `{SYSTEM}` field, and just put plain text.

```python
chat_template = '''{SYSTEM}
USER: {INPUT}
ASSISTANT: {OUTPUT}'''
```

Use below if you want to use the Llama-3 prompt format. You must use the `instruct` and not the `base` model if you use this!
```python
chat_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>'''
```

For the ChatML format:
```python
chat_template = '''<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>'''
```

The issue is the Alpaca format has 3 fields, whilst OpenAI style chatbots must only use 2 fields (instruction and response). That's why we used the `to_sharegpt` function to merge these columns into 1.
"""

chat_template = """{SYSTEM}
USER: {INPUT}
ASSISTANT: {OUTPUT}"""

from unsloth import apply_chat_template

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)

import wandb
import os

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "llama-3.1-financial-assistant"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

"""<a name="Train"></a>
### Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

#TODO rodar esta merda localmente
"""

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=1,
        # gradient_accumulation_steps = 4, this is batch size copium never use
        warmup_steps=5,
        # max_steps=None,
        num_train_epochs=2,  # For longer training runs!
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
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
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
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
pass

"""<a name="Ollama"></a>
### Ollama Support

[Unsloth](https://github.com/unslothai/unsloth) now allows you to automatically finetune and create a [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md), and export to [Ollama](https://ollama.com/)! This makes finetuning much easier and provides a seamless workflow from `Unsloth` to `Ollama`!

Let's first install `Ollama`!
"""

"""Next, we shall save the model to GGUF / llama.cpp

We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
* `q8_0` - Fast conversion. High resource use, but generally acceptable.
* `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
* `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

We also support saving to multiple GGUF options in a list fashion! This can speed things up by 10 minutes or more if you want multiple export formats!
"""

# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False:
    model.push_to_hub_gguf(
        "hf/model", tokenizer, quantization_method="q4_k_m", token=""
    )

# Save to multiple GGUF options - much faster if you want multiple!
if True:
    model.push_to_hub_gguf(
        "rmeireles/financial-llama-assistant",
        tokenizer,
        quantization_method=[
            "q4_k_m",
            "q8_0",
            "q5_k_m",
        ],
        token="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",  # Get a token at https://huggingface.co/settings/tokens
    )
