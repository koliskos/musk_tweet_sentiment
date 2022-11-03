import datasets
from transformers import RobertaTokenizer
from transformers import  RobertaForMultipleChoice

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/aristo-roberta")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/aristo-roberta")
dataset = datasets.load_dataset(
    "arc",
    split=["train", "validation", "test"],
)
training_examples = dataset[0]
evaluation_examples = dataset[1]
test_examples = dataset[2]

example=training_examples[0]
print(example)
example_id = example["example_id"]
print(example_id)
question = example["question"]
print(question)
label_example = example["answer"]
print(label_example)
options = example["options"]
print(options)

if label_example in ["A", "B", "C", "D", "E"]:
    label_map = {label: i for i, label in enumerate(
                    ["A", "B", "C", "D", "E"])}
elif label_example in ["1", "2", "3", "4", "5"]:
    label_map = {label: i for i, label in enumerate(
                    ["1", "2", "3", "4", "5"])}
else:
    print(f"{label_example} not found")
while len(options) < 5:
    empty_option = {}
    empty_option['option_context'] = ''
    empty_option['option_text'] = ''
    options.append(empty_option)
choices_inputs = []
for ending_idx, option in enumerate(options):
    ending = option["option_text"]
    context = option["option_context"]
    if question.find("_") != -1:
        # fill in the banks questions
        question_option = question.replace("_", ending)
    else:
        question_option = question + " " + ending

    inputs = tokenizer(
        context,
        question_option,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=False,
    )

    if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
        logging.warning(f"Question: {example_id} with option {ending_idx} was truncated")
    choices_inputs.append(inputs)
label = label_map[label_example]
input_ids = [x["input_ids"] for x in choices_inputs]
attention_mask = (
    [x["attention_mask"] for x in choices_inputs]
     # as the senteces follow the same structure, just one of them is
     # necessary to check
    if "attention_mask" in choices_inputs[0]
    else None
)
example_encoded = {
    "example_id": example_id,
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
    "label": label
}
output = model(**example_encoded)
print(output)
print(output['label'])
