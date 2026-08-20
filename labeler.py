from openai import OpenAI

class ThemeLabeler:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_label(self, cluster_id, samples):
        if cluster_id == -1:
            return "Unclassified - High Novelty Score"

        samples_text = "\n".join([f"- {s}" for s in samples])
        prompt = (
            "You are a customer support analyst. Given the following customer complaints belonging "
            "to the same category, output ONLY a concise, uppercase 3-to-5-word business category theme.\n\n"
            f"Complaints:\n{samples_text}\n\nTheme:"
        )

        response = self.client.chat.completions.create(
            model="local-model",  # LM Studio uses whichever model is active
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()