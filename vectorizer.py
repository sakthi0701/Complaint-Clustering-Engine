import numpy as np
import pandas as pd
from openai import OpenAI

class LMStudioVectorizer:
    def __init__(self, base_url="http://127.0.0.1:1234/v1", api_key="sk-lm-FS20F6sI:xtIz6WGItyAMVY2ZwpBo"):
        # LM Studio exposes an OpenAI-compatible API at /v1
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # Model identifier as loaded in LM Studio
        self.model_name = "all-MiniLM-L6-v2"

    def clean_and_filter(self, df, text_column='complaint_text'):
        """Filters out empty strings or short texts (< 3 words)."""
        df = df.dropna(subset=[text_column])
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        return df[df['word_count'] >= 3].copy()

    import pandas as pd

    def vectorize(self, text_list):
        # 1. Convert pandas Series to list if necessary
        if isinstance(text_list, pd.Series):
            text_list = text_list.tolist()
        elif hasattr(text_list, "tolist"):
            text_list = text_list.tolist()

        embeddings = []
        
        # 2. Loop through each complaint individually
        for text in text_list:
            # Clean line breaks just like the documentation sample
            clean_text = str(text).replace("\n", " ")
            
            # Call the API using the exact documentation structure
            response = self.client.embeddings.create(
                input=[clean_text],  # Kept inside a single-element list
                model=self.model_name,
                encoding_format="float"
            )
            
            # Safely extract the embedding array
            if response.data and len(response.data) > 0:
                vector = response.data[0].embedding
                embeddings.append(vector)
            else:
                raise ValueError(f"LM Studio returned no embedding data for text: '{clean_text}'")
                
        return embeddings
