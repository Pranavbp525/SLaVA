import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import \
    (AutoTokenizer,
     CLIPImageProcessor,
     AutoModelForCausalLM,
     LlavaForConditionalGeneration)

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


def save_checkpoint(model, optimizer, step):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("Checkpoint saved at:", checkpoint_path)


def load_latest_checkpoint(model, optimizer):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print("Latest checkpoint loaded from:", checkpoint_path)
        return step
    else:
        print("No checkpoints found.")
        return 0


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
    dataset = dataset.map(process_and_add_input, batched=True)

    llava = LlavaForConditionalGeneration.from_pretrained(llava_model_name)

    for param in llava.vision_tower.parameters():
        param.requires_grad = False

    llava.multi_modal_projector = MLP()
    llava.language_model = None

    language_model = AutoModelForCausalLM.from_pretrained(
        language_model
    )

    llava.language_model = language_model

    llava.config.image_token_index = 256000
    num_added_tokens = processor.tokenizer.add_tokens('<image>')
    print("Added", num_added_tokens, "tokens")
    print("New token `<image>` has ID:", processor.tokenizer.convert_tokens_to_ids('<image>'))
    llava.resize_token_embeddings(len(processor.tokenizer))

    for param in llava.language_model.parameters():
        param.requires_grad = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    llava.to(device)
    optimizer = torch.optim.AdamW(llava.parameters(), lr=2e-4)

    data_collator = SLavaDataCollator(processor, image_path)
    optimizer = torch.optim.AdamW(llava.parameters(), lr=2e-4)

    train_dataloader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=1, num_workers=4)

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)

    progress_bar = tqdm(range(num_training_steps))

    llava.train()

    os.makedirs(checkpoint_dir, exist_ok=True)

    step = load_latest_checkpoint(llava, optimizer)
    for epoch in range(num_epochs):
        for batch_step, batch in enumerate(train_dataloader, start=step):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = llava(**batch)
            loss = outputs.loss

            optimizer.step()
            optimizer.zero_grad()

            if batch_step % 10 == 0:  # Update every 10 steps
                print(f"Epoch {epoch + 1}, Step {batch_step}, Loss: {loss.item():.4f}")

            if (batch_step + 1) % 10000 == 0:  # Save checkpoint
                save_checkpoint(llava, optimizer, batch_step + 1)
                llava.push_to_hub("llava-pretrain")
                processor.push_to_hub("llava-pretrain")

            progress_bar.update(1)

    torch.save(llava.multi_modal_projector.state_dict(), 'projector_weights.pth')


if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    dataset_path = "/kaggle/input/pretrain-chat-new/pretrain-chat.json"
    llava_model_name = "liuhaotian/llava-v1.5-7b"
    language_model = "google/gemma-1.1-2b-it"
    vision_tower = "openai/clip-vit-large-patch14-336"
    image_path = "/kaggle/input/llava-cc3m-pretrain-595k"

    train(dataset_path, llava_model_name, language_model, vision_tower, image_path)
