import torch
from datasets import load_dataset
from peft import \
    (LoraConfig,
     get_peft_model,
     prepare_model_for_kbit_training)
from transformers import \
    (AutoTokenizer,
     TrainingArguments,
     BitsAndBytesConfig,
     CLIPImageProcessor,
     AutoModelForCausalLM,
     LlavaForConditionalGeneration)
from trl import SFTTrainer

from datacollator import SLavaDataCollator
from processor import SlavaProcessor


class MLP(torch.nn.Module):
    def __init__(self, in_features=1024, out_features=2048):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, out_features, bias=True)
        self.gelu = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(out_features, out_features, bias=True)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


def process_and_add_input(batch):
    role_mapping = {'human': 'user', 'gpt': 'assistant'}
    batched_adapted_conversations = []
    for conversations in batch['conversations']:
        adapted_conversations = []
        for conv in conversations:
            adapted_conversations.append({
                'role': role_mapping[conv['from']],
                'content': conv['value']
            })
        batched_adapted_conversations.append(adapted_conversations)

    batch['conversations'] = batched_adapted_conversations
    return batch


def train(dataset_path, llava_model_name, language_model, vision_tower, image_path):
    processor = SlavaProcessor(
        tokenizer=AutoTokenizer.from_pretrained(language_model),
        image_processor=CLIPImageProcessor.from_pretrained(vision_tower)
    )

    dataset = load_dataset("json", data_files=dataset_path)

    dataset = dataset["train"]
    train_test_split = dataset.train_test_split(test_size=0.01)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    train_dataset = train_dataset.map(process_and_add_input, batched=True)
    val_dataset = val_dataset.map(process_and_add_input, batched=True)

    llava = LlavaForConditionalGeneration.from_pretrained(llava_model_name)

    for param in llava.vision_tower.parameters():
        param.requires_grad = False

    llava.multi_modal_projector = MLP()
    llava.multi_modal_projector.load_state_dict(torch.load('projector_weights.pth'))
    llava.language_model = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    language_model = AutoModelForCausalLM.from_pretrained(
        language_model,
        quantization_config=bnb_config
    )

    language_model = prepare_model_for_kbit_training(language_model)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'q_proj',
            'v_proj'
        ]
    )

    language_model = get_peft_model(language_model, peft_config)
    llava.language_model = language_model

    llava.config.image_token_index = 256000
    num_added_tokens = processor.tokenizer.add_tokens('<image>')
    print("Added", num_added_tokens, "tokens")
    print("New token `<image>` has ID:", processor.tokenizer.convert_tokens_to_ids('<image>'))
    llava.resize_token_embeddings(len(processor.tokenizer))

    data_collator = SLavaDataCollator(processor, image_path)
    optimizer = torch.optim.AdamW(llava.parameters(), lr=2e-4)

    training_arguments = TrainingArguments(
        output_dir="./slava",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        fp16_full_eval=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        logging_strategy="steps",
        learning_rate=2e-4,
        push_to_hub=True,
        save_total_limit=2,
        resume_from_checkpoint=True
    )

    # revise this
    trainer = SFTTrainer(
        model=llava,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        args=training_arguments,
        peft_config=peft_config,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
        optimizers=(optimizer, None)
    )

    trainer.train()

    trainer.push_to_hub("Fine-tuned Model")


if __name__ == "__main__":
    dataset_path = "/kaggle/input/llava-instruct-80k/llava_instruct_80k.json"
    llava_model_name = "liuhaotian/llava-v1.5-7b"
    language_model = "google/gemma-1.1-2b-it"
    vision_tower = "openai/clip-vit-large-patch14-336"
    image_path = "/kaggle/input/llava-instruct-150k-images/train2017"

    train(dataset_path, llava_model_name, language_model, vision_tower, image_path)
