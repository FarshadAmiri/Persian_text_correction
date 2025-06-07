import os
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from typing import Any, Dict, List, Union

os.chdir(r"D:\Git_repos\Persian_text_correction\datasets")

# Model & Processor

# model_name = "openai/whisper-medium"
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

df = pd.read_excel(r"D:\Git_repos\Persian_text_correction\datasets\myaudio_tiny\myaudio_tiny.xlsx")
df = df.rename(columns={"audio": "audio_path", "text": "transcription"})

print(df.columns) 

df.head


def prepare_example(batch):
    audio = batch["audio_path"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    input_features = inputs.input_features[0]
    labels = processor.tokenizer(batch["transcription"], return_tensors="pt").input_ids[0]
    return {"input_features": input_features, "labels": labels}

os.chdir(r"D:\Git_repos\Persian_text_correction\datasets\myaudio_tiny")
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio_path", Audio())
dataset = dataset.map(prepare_example)


# LoRA setup
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="D:\Git_repos\Persian_text_correction\models\whisper-medium-ft-fa",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    logging_steps=25,
    save_steps=200,
    eval_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none"
)

# 8. Trainer
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor  # Optional
)

# 9. ✅ Train
trainer.train()