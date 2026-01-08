from app.models.bert_model import BertModel
from app.api.dto.bert_response import BertClassificationResponse
from app.core.config import settings
from typing import List


class BertService:

    def __init__(self):
        self.model = BertModel.load()

    async def classify(self, text: str) -> BertClassificationResponse:
        result = self.model.classify(text)

        predicted_class = result["predicted_class"]
        label = settings.LABEL_MAP.get(predicted_class, f"class_{predicted_class}")

        class_probabilities = {
            settings.LABEL_MAP.get(i, f"class_{i}"): float(prob)
            for i, prob in enumerate(result["probabilities"])
        }

        return BertClassificationResponse(
            text=text,
            predicted_label=label,
            predicted_class=predicted_class,
            confidence=float(result["confidence"]),
            class_probabilities=class_probabilities,
            meta={
                "model": "bert-finetuned",
                "num_labels": settings.NUM_LABELS,
            },
        )

    async def batch_classify(self, texts: List[str]) -> List[BertClassificationResponse]:
        results = []
        for text in texts:
            result = await self.classify(text)
            results.append(result)
        return results
