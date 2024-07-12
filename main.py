from datasets import Dataset
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments


data = {
    'queries': [
        "Show me all open issues in project ABC",
        "What are the high priority bugs assigned to me?",
        "List all tasks due by end of this month"
    ],
    'jql': [
        "project = ABC AND status = Open",
        "assignee = currentUser() AND priority = High AND issuetype = Bug",
        "duedate <= endOfMonth() AND issuetype = Task"
    ]
}

dataset = Dataset.from_dict(data)

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_function(examples):
    inputs = examples['queries']
    targets = examples['jql']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained('t5-small')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

def translate_to_jql(query):
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids)
    jql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jql_query

user_query = "Show me all open issues in project ABC"
jql_query = translate_to_jql(user_query)
print(jql_query)

