"""
Explanation Generator for Baseline LLM Mapping Framework.
Generates analytical rationale and explanation text for mapped disclosure pairs.
"""

from typing import Optional
from llm_mapping.data.brsr_loader import BRSRDisclosure
from llm_mapping.data.gri_loader import GRIDisclosure
from llm_mapping.llm.ollama_client import OllamaClient
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("explanation")


class ExplanationGenerator:
    """Utility class to generate natural language explanations for mapped pairs."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()

    def generate_explanation(
        self,
        brsr: BRSRDisclosure,
        gri: GRIDisclosure,
        relationship: str = "Close Match",
    ) -> str:
        """
        Generates a natural language explanation for why a BRSR disclosure aligns with a GRI disclosure.

        Args:
            brsr: BRSR disclosure object.
            gri: GRI disclosure object.
            relationship: Alignment relationship tag.

        Returns:
            str: Generated explanation text.
        """
        prompt = (
            f"Explain in detail why BRSR Disclosure '{brsr.id}' ({brsr.label}) aligns with GRI Disclosure '{gri.id}' "
            f"({gri.label}) under a '{relationship}' relationship.\n"
            f"BRSR Text: {brsr.text}\n"
            f"GRI Text: {gri.text}\n"
            f"Provide a concise analytical explanation."
        )

        response = self.llm_client.generate(prompt=prompt)
        if response and response.text:
            return response.text

        # Fallback template explanation
        return (
            f"BRSR disclosure '{brsr.id}' ({brsr.label}) was aligned with GRI disclosure '{gri.id}' "
            f"({gri.label}) under a '{relationship}' relationship based on semantic vector similarity "
            f"and LLM disclosure requirement matching."
        )
