from huggingface_hub import notebook_login

notebook_login()

# MODEL = "openai/whisper-small"
MODEL = "openai/whisper-medium"
# MODEL = "openai/whisper-large"

# DATASET_NAME = "mozilla-foundation/common_voice_11_0"
DATASET_NAME = "mozilla-foundation/common_voice_12_0"
LANGUAGE = "fa"


PARTIAL_DATASET = False   # True or False for debugging\

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

if PARTIAL_DATASET:
    common_voice = DatasetDict({
        "train": load_dataset(DATASET_NAME, LANGUAGE, split=f"train[:1700]", trust_remote_code=True),
        "validation": load_dataset(DATASET_NAME, LANGUAGE, split="validation[:200]", trust_remote_code=True)
    })
else:
    common_voice = DatasetDict({
        "train": load_dataset(DATASET_NAME, LANGUAGE, split=f"train", trust_remote_code=True),
        "validation": load_dataset(DATASET_NAME, LANGUAGE, split="validation[:400]", trust_remote_code=True)
    })

train_dataset = common_voice["train"]
eval_dataset = common_voice["validation"]

# common_voice["test"] = load_dataset(DATASET_NAME, "fa", split="test", use_auth_token=True)

# print(common_voice)

from datasets import Audio

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))