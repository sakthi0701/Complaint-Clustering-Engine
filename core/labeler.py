import os
import pandas as pd
from groq import Groq


class ThemeLabeler:
    """
    Generates 3-5-word business category theme labels for complaint clusters.

    Uses Groq API with model: openai/gpt-oss-120b.
    Note: reasoning_effort is intentionally NOT set for label generation.
    Simple classification does not benefit from chain-of-thought reasoning,
    and enabling it with a low max_completion_tokens budget causes the model
    to exhaust the token limit on thinking, returning an empty label.
    """

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"

    # ------------------------------------------------------------------
    # Sample extraction
    # ------------------------------------------------------------------

    def get_cluster_samples(
        self,
        group_df: pd.DataFrame,
        text_column: str = "complaint_text",
        top_n: int = 3,
    ) -> list[str]:
        """Returns up to top_n representative complaint texts from a cluster."""
        return group_df[text_column].dropna().head(top_n).tolist()

    # ------------------------------------------------------------------
    # LLM Labeling via Groq
    # ------------------------------------------------------------------

    def generate_label(self, cluster_id: int, samples: list[str]) -> str:
        """
        Calls the Groq API with streaming and buffers the response into a
        clean 3-5-word uppercase theme label.

        Args:
            cluster_id: The numeric cluster ID (-1 for sandbox).
            samples:    List of representative complaint strings.

        Returns:
            A concise uppercase theme string, e.g. "MOBILE APP CRASHES".
        """
        if cluster_id == -1:
            return "Unclassified — High Novelty Score"

        if not samples:
            return f"Cluster {cluster_id}"

        samples_text = "\n".join([f"- {s}" for s in samples])
        prompt = (
            "You are a senior customer support analyst working at a bank or fintech company.\n"
            "Given the following customer complaints that all belong to the SAME issue category, "
            "output ONLY a concise, UPPERCASE, 3-to-5-word business department theme.\n"
            "Do not add any punctuation, explanation, or extra words. Output the theme directly.\n\n"
            f"Complaints:\n{samples_text}\n\nTheme:"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=256,   # Must be high enough for model output
                top_p=1,
                stream=True,
                stop=None,
                # reasoning_effort intentionally omitted — simple classification
                # task does not need chain-of-thought, and enabling it exhausts
                # max_completion_tokens before the model outputs the label.
            )

            # Collect streamed chunks into a single label string
            label_parts: list[str] = []
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    label_parts.append(delta)

            label = "".join(label_parts).strip()
            # Fallback if something still goes wrong
            return label if label else f"Cluster {cluster_id}"

        except Exception as exc:
            print(f"[ThemeLabeler] Groq API error for cluster {cluster_id}: {exc}")
            return f"Cluster {cluster_id}"
