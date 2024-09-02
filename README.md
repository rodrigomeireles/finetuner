# Finetuner

Finetuner is a Python tool for fine-tuning language models using the `unsloth` library. It allows you to load a pretrained model, prepare a dataset, train the model, and save the fine-tuned model with various quantization options.

## Usage

To use Finetuner, run the script from the command line with the required arguments:

```bash
uv run finetuner.py \
    --model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --dataset_name="britojr/sec10q_v2" \
    --prompt_format_path="path/to/prompt_format.txt" \
    --chat_format_path="path/to/chat_format.txt" \
    --batch_size=2 \
    --hf_repo="your_name/fine_tuned_model"
