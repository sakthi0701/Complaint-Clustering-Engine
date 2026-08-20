import numpy as np
import pandas as pd
from openai import OpenAI


class LMStudioVectorizer:
    """
    Embeds complaint texts using the LM Studio OpenAI-compatible endpoint.
    Model: all-MiniLM-L6-v2 loaded in LM Studio.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "sk-lm-FS20F6sI:xtIz6WGItyAMVY2ZwpBo",
    ):
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = "all-MiniLM-L6-v2"

    def clean_and_filter(self, df: pd.DataFrame, text_column: str = "complaint_text") -> pd.DataFrame:
        """Drops nulls and texts shorter than 3 words."""
        df = df.dropna(subset=[text_column]).copy()
        df["word_count"] = df[text_column].apply(lambda x: len(str(x).split()))
        return df[df["word_count"] >= 3].copy()

    def vectorize(self, text_list) -> list[list[float]]:
        """Embeds a list of complaint texts. Returns a list of float vectors."""
        if isinstance(text_list, pd.Series):
            text_list = text_list.tolist()
        elif hasattr(text_list, "tolist"):
            text_list = text_list.tolist()

        embeddings = []
        for text in text_list:
            clean_text = str(text).replace("\n", " ")
            response = self.client.embeddings.create(
                input=[clean_text],
                model=self.model_name,
                encoding_format="float",
            )
            if response.data and len(response.data) > 0:
                embeddings.append(response.data[0].embedding)
            else:
                raise ValueError(f"LM Studio returned no embedding for: '{clean_text}'")

        return embeddings
