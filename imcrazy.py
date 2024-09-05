chat_template_unsloth = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""


with open("chat_format.txt", "r") as file:
    chat_template = file.read()[:-1]
print(chat_template)
print("Unsloth")
print(chat_template_unsloth)
assert chat_template == chat_template_unsloth
