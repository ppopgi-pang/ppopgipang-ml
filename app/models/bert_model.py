from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.core.config import settings


class BertModel:
    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def load(cls):
        if cls._instance is None:
            cls._instance = cls()
            print(f"Loading BERT model from {settings.BERT_MODEL_PATH}...")

            cls._instance._tokenizer = AutoTokenizer.from_pretrained(
                settings.BERT_MODEL_PATH,
                local_files_only=True,
            )

            cls._instance._model = AutoModelForSequenceClassification.from_pretrained(
                settings.BERT_MODEL_PATH,
                local_files_only=True,
                use_safetensors=True,
            )

            if torch.cuda.is_available():
                cls._instance._model = cls._instance._model.cuda()
                print("BERT model loaded on GPU")
            else:
                print("BERT model loaded on CPU")

            cls._instance._model.eval()

        return cls._instance

    def classify(self, text: str):
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        return {
            "predicted_class": predicted_class.item(),
            "probabilities": probabilities.cpu().numpy()[0].tolist(),
            "confidence": probabilities.max().item(),
        }
