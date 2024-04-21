import os
from PIL import Image


# Code adapted from vsft_llava.py script of the Huggingface examples script
class SLavaDataCollator:
    def __init__(self, processor, image_path):
        self.processor = processor
        self.image_path = image_path

    def __call__(self, examples):

        texts = []
        images = []
        for example in examples:
            messages = example["conversations"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(Image.open(
                os.path.join(self.image_path, example['image'])).convert("RGB"))

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
