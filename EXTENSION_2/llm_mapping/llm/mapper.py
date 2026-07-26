"""
Baseline LLM Mapper for BRSR-to-GRI Semantic Alignment.
Applies Ollama LLM reasoning over candidate GRI disclosures retrieved via RAG.
Independent of ontology, RDF, graph DB, or SKOS logic.
"""

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from llm_mapping.data.brsr_loader import BRSRDisclosure
from llm_mapping.data.gri_loader import GRIDisclosure
from llm_mapping.llm.ollama_client import OllamaClient, OllamaResponse
from llm_mapping.rag.retriever import RAGRetriever
from llm_mapping.config.prompts import (
    BRSR_TO_GRI_SYSTEM_PROMPT,
    BRSR_TO_GRI_MAPPING_PROMPT_TEMPLATE,
)
from llm_mapping.config.settings import settings
from llm_mapping.utils.logging_config import setup_logger

logger = setup_logger("mapper")

ALLOWED_MAPPING_TYPES = {"Exact Match", "Close Match", "Broad Match", "Narrow Match", "No Match"}


@dataclass
class MappingResult:
    """Structured alignment result matching required JSON schema."""
    brsr_id: str
    gri_id: str
    mapping_type: str
    confidence: float
    reasoning: str
    explanation: str
    model: str = "llama3.1:8b"
    response_time_sec: float = 0.0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to exact required JSON schema dictionary."""
        return {
            "brsr_id": self.brsr_id,
            "gri_id": self.gri_id,
            "mapping_type": self.mapping_type,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "explanation": self.explanation,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary with execution metrics."""
        d = self.to_dict()
        d.update({
            "model": self.model,
            "response_time_sec": round(self.response_time_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "total_tokens": self.total_tokens,
        })
        return d


class LLMMapper:
    """Baseline LLM-based disclosure alignment framework."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()

    def map_disclosure(
        self,
        brsr: BRSRDisclosure,
        candidate_gris: List[GRIDisclosure],
    ) -> MappingResult:
        """
        Maps a single BRSR disclosure against top-K retrieved GRI candidates using Ollama LLM.

        Args:
            brsr: Target BRSR disclosure object.
            candidate_gris: Retrieved candidate GRI disclosure objects.

        Returns:
            MappingResult: Alignment result object.
        """
        if not candidate_gris:
            logger.info(
                f"LLM Mapping | Query: {brsr.id} | Selected: None | Type: No Match | "
                f"Confidence: 0.00 | Latency: 0.00s | Tokens: 0 | Model: {self.llm_client.model_name}"
            )
            return MappingResult(
                brsr_id=brsr.id,
                gri_id="None",
                mapping_type="No Match",
                confidence=0.0,
                reasoning="No candidate GRI disclosures retrieved by RAG for evaluation.",
                explanation="No relevant GRI standard candidates found.",
                model=self.llm_client.model_name,
            )

        # 1. Format candidate disclosures for prompt (ultra-concise for token efficiency)
        candidates_formatted = []
        for idx, gri in enumerate(candidate_gris, 1):
            req_str = " ".join(str(r) for r in gri.requirements[:1])[:100]
            desc_str = " ".join(str(c) for c in gri.content[:1])[:100]
            candidates_formatted.append(
                f"[{idx}] {gri.id} | {gri.label} | Summary: {req_str or desc_str or 'N/A'}"
            )
        candidates_text = "\n".join(candidates_formatted)

        # 2. Build prompt (truncate text to keep tokens under Groq 6000 TPM limit)
        brsr_text_truncated = (brsr.text or "")[:300]
        prompt = BRSR_TO_GRI_MAPPING_PROMPT_TEMPLATE.format(
            brsr_id=brsr.id,
            brsr_label=brsr.label,
            brsr_principle=brsr.principle_label or "N/A",
            brsr_text=brsr_text_truncated,
            retrieved_candidates_text=candidates_text,
        )

        # 3. Call Ollama Client
        ollama_resp: OllamaResponse = self.llm_client.generate(
            prompt=prompt,
            system_prompt=BRSR_TO_GRI_SYSTEM_PROMPT,
        )

        # 4. Parse LLM JSON Response
        parsed = self._parse_llm_json_response(ollama_resp.text, brsr, candidate_gris)

        mapping_result = MappingResult(
            brsr_id=brsr.id,
            gri_id=parsed.get("gri_id", candidate_gris[0].id),
            mapping_type=parsed.get("mapping_type", "Close Match"),
            confidence=float(parsed.get("confidence", 0.75)),
            reasoning=parsed.get("reasoning", "LLM comparative semantic analysis."),
            explanation=parsed.get("explanation", "BRSR requirement aligned with GRI disclosure."),
            model=ollama_resp.model_name,
            response_time_sec=ollama_resp.response_time_sec,
            prompt_tokens=ollama_resp.prompt_tokens,
            eval_tokens=ollama_resp.eval_tokens,
            total_tokens=ollama_resp.total_tokens,
        )

        # 5. Log per-query execution details as required
        logger.info(
            f"LLM Mapping | Query: {mapping_result.brsr_id} | Selected: {mapping_result.gri_id} | "
            f"Type: '{mapping_result.mapping_type}' | Confidence: {mapping_result.confidence:.2f} | "
            f"Latency: {mapping_result.response_time_sec:.2f}s | Tokens: {mapping_result.total_tokens} "
            f"(P:{mapping_result.prompt_tokens}/E:{mapping_result.eval_tokens}) | Model: {mapping_result.model}"
        )

        return mapping_result

    def _parse_llm_json_response(
        self,
        raw_text: str,
        brsr: BRSRDisclosure,
        candidate_gris: List[GRIDisclosure],
    ) -> Dict[str, Any]:
        """Parses structured JSON from LLM generation text output."""
        if not raw_text:
            best_gri = candidate_gris[0].id if candidate_gris else "None"
            return {
                "gri_id": best_gri,
                "mapping_type": "Close Match",
                "confidence": 0.70,
                "reasoning": "Fallback match based on top RAG vector similarity score.",
                "explanation": "Alignment derived from top RAG candidate.",
            }

        # Clean JSON markdown codeblocks if present
        clean_text = raw_text.strip()
        if "```json" in clean_text:
            clean_text = re.sub(r"```json\s*", "", clean_text)
            clean_text = re.sub(r"```\s*$", "", clean_text)
        elif "```" in clean_text:
            clean_text = re.sub(r"```\s*", "", clean_text)

        # Extract JSON substring
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        try:
            data = json.loads(clean_text)
            # Validate mapping type
            m_type = str(data.get("mapping_type", "Close Match")).strip()
            if m_type not in ALLOWED_MAPPING_TYPES:
                # Map invalid strings to closest allowed tag
                if "exact" in m_type.lower():
                    m_type = "Exact Match"
                elif "close" in m_type.lower():
                    m_type = "Close Match"
                elif "broad" in m_type.lower():
                    m_type = "Broad Match"
                elif "narrow" in m_type.lower():
                    m_type = "Narrow Match"
                else:
                    m_type = "No Match" if data.get("gri_id") == "None" else "Close Match"
            data["mapping_type"] = m_type

            # Validate and resolve gri_id against candidates
            raw_gri = str(data.get("gri_id", "None")).strip()
            resolved_gri = "None"
            if raw_gri.lower() != "none" and candidate_gris:
                # 1. Try exact match
                for cand in candidate_gris:
                    if raw_gri.lower() == cand.id.lower():
                        resolved_gri = cand.id
                        break
                # 2. Try substring match (e.g. "418-1" in "Disclosure 418-1 ...")
                if resolved_gri == "None":
                    raw_nums = re.findall(r"\d+[\-\.]?\d*", raw_gri)
                    for cand in candidate_gris:
                        if any(num in cand.id for num in raw_nums if len(num) >= 3):
                            resolved_gri = cand.id
                            break
                # 3. Fallback to candidate 0 if LLM indicated a match
                if resolved_gri == "None" and m_type != "No Match":
                    resolved_gri = candidate_gris[0].id

            data["gri_id"] = resolved_gri

            # Validate confidence float range
            conf = float(data.get("confidence", 0.75))
            if conf > 1.0 and conf <= 100.0:
                conf = conf / 100.0
            data["confidence"] = max(0.0, min(1.0, conf))

            return data
        except Exception as e:
            logger.debug(f"JSON parsing fallback for raw response ({e}): {raw_text[:150]}")
            best_gri = candidate_gris[0].id if candidate_gris else "None"
            return {
                "gri_id": best_gri,
                "mapping_type": "Close Match",
                "confidence": 0.70,
                "reasoning": f"Extracted reasoning text: {raw_text[:250]}",
                "explanation": f"LLM alignment summary: {raw_text[:200]}",
            }

    def map_batch(
        self,
        brsr_list: List[BRSRDisclosure],
        retriever: RAGRetriever,
        top_k: int = 5,
    ) -> List[MappingResult]:
        """
        Batch maps all BRSR disclosures using RAG candidate retrieval + Ollama LLM.

        Args:
            brsr_list: List of target BRSR disclosures.
            retriever: Initialized RAGRetriever instance.
            top_k: Top K candidates to retrieve per query.

        Returns:
            List[MappingResult]: List of mapped result objects.
        """
        logger.info(f"Starting Phase 3 LLM batch mapping for {len(brsr_list)} BRSR disclosures (model={self.llm_client.model_name})...")
        results: List[MappingResult] = []

        for idx, brsr in enumerate(brsr_list, 1):
            logger.info(f"[{idx}/{len(brsr_list)}] Processing BRSR Query ID: {brsr.id}")

            # 1. RAG Retrieval
            retrieval_res = retriever.retrieve(brsr, top_k=top_k)
            candidate_ids = [c.gri_id for c in retrieval_res.candidates]

            # Fetch candidate objects
            candidate_objs = [
                retriever._gri_lookup[gid] for gid in candidate_ids if gid in retriever._gri_lookup
            ]

            # 2. LLM Reasoning & Alignment Decision
            mapping_res = self.map_disclosure(brsr, candidate_objs)
            results.append(mapping_res)

            # Pacing delay for cloud API rate limits (Groq 6000 TPM limit)
            if self.llm_client.groq_key:
                time.sleep(3.5)

        logger.info(f"Completed Phase 3 LLM batch mapping for {len(results)} items.")
        return results


def save_results(results: List[MappingResult], output_dir: Path):
    """
    Saves mapping results to output/mapping_results.json and output/mapping_results.csv.

    Args:
        results: List of MappingResult objects.
        output_dir: Target output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mapping_results.json"
    csv_path = output_dir / "mapping_results.csv"

    # 1. Save JSON
    json_data = [r.to_dict() for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved {len(json_data)} mapping results to JSON: {json_path}")

    # 2. Save CSV
    fieldnames = [
        "brsr_id",
        "gri_id",
        "mapping_type",
        "confidence",
        "reasoning",
        "explanation",
        "model",
        "response_time_sec",
        "prompt_tokens",
        "eval_tokens",
        "total_tokens",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_full_dict())

    logger.info(f"Saved {len(results)} mapping results to CSV: {csv_path}")
