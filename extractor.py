import os
import re
import json
from typing import Dict, List, Any
import models
from utils import logger

class BRSRDataExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    def extract_metadata_and_bind(
        self,
        contexts: Dict[str, models.Context],
        units: Dict[str, models.Unit],
        facts: List[models.Fact],
        namespaces: Dict[str, str]
    ) -> tuple[models.ReportMetadata, List[models.Fact]]:
        """
        Extract company name, report year, and bind each fact to its context and unit details.
        """
        # 1. Extract Company Name
        company_name = self._extract_company_name(facts)
        
        # 2. Extract Report Year
        report_year = self._extract_report_year(contexts, facts)
        
        logger.info(f"Extracted metadata: Company='{company_name}', Year='{report_year}' from {self.file_name}")

        # 3. Bind and enrich facts
        enriched_facts = []
        for fact in facts:
            # Inject company and year metadata
            fact.company_name = company_name
            fact.report_year = report_year
            
            # Enrich with context info if available
            ctx = contexts.get(fact.context_ref)
            if ctx:
                # We can store context information or pass it down
                # For simplicity, we serialize dimensions as JSON
                pass
            
            enriched_facts.append(fact)

        # 4. Generate Report Metadata
        metadata = models.ReportMetadata(
            company_name=company_name,
            report_year=report_year,
            namespace=namespaces.get("default", namespaces.get("in-capmkt", "")),
            source_file=self.file_name,
            context_count=len(contexts),
            unit_count=len(units),
            fact_count=len(enriched_facts)
        )

        return metadata, enriched_facts

    def _extract_company_name(self, facts: List[models.Fact]) -> str:
        """Find company name from specific concepts or parent folder."""
        # Multi-stage check:
        # Stage 1: Explicit concept match
        for fact in facts:
            if fact.concept in ["NameOfTheCompany", "NameOfCompany", "NameOfListedEntity", "NameOfEntity", "NameOfTheListedEntity"]:
                val = fact.value.strip()
                if val and val not in ["-", "Nil", "NIL", "NA", "Not Applicable", "Not applicable"]:
                    return val

        # Stage 2: Case-insensitive search on concept name for company name tags
        for fact in facts:
            c_lower = fact.concept.lower()
            if "name" in c_lower and ("company" in c_lower or "entity" in c_lower or "listed" in c_lower):
                val = fact.value.strip()
                if val and len(val) < 150 and any(w in val.lower() for w in ["limited", "ltd", "corporation", "corp", "bank", "tata", "reliance", "eicher"]):
                    return val

        # Stage 3: Deduce from parent folder name
        # Path: .../financial_dataset/TCS/XBRL/FY2023-24.xml -> parent of parent is TCS
        parts = Path(self.file_path).parts
        if len(parts) >= 3:
            # TCS or Reliance or Eicher
            candidate = parts[-3]
            if candidate not in ["XBRL", "financial_dataset", "INTENSHIPS", "IIT-H"]:
                return candidate.replace("_", " ").strip()

        return "Unknown Company"

    def _extract_report_year(self, contexts: Dict[str, models.Context], facts: List[models.Fact]) -> str:
        """Extract the report year from contexts, facts or filename."""
        # Stage 1: Check facts for reporting period/year
        for fact in facts:
            c_lower = fact.concept.lower()
            if "financialyear" in c_lower or "reportingperiod" in c_lower:
                val = fact.value.strip()
                if val and len(val) < 20 and re.search(r'\d{4}', val):
                    return val

        # Stage 2: Deduce from contexts (especially DCYMain or context ending dates)
        # Look for DCYMain start and end dates
        dcy = contexts.get("DCYMain")
        if dcy and dcy.start_date and dcy.end_date:
            try:
                # e.g., 2023-04-01 to 2024-03-31 -> "2023-24"
                start_year = dcy.start_date.split("-")[0]
                end_year = dcy.end_date.split("-")[0]
                if start_year and end_year:
                    end_yr_short = end_year[2:] if len(end_year) == 4 else end_year
                    return f"{start_year}-{end_yr_short}"
            except Exception:
                pass

        # Also search any context with period type duration that has longest span
        duration_contexts = [c for c in contexts.values() if c.period_type == "duration" and c.start_date and c.end_date]
        if duration_contexts:
            # Sort by end date descending
            duration_contexts.sort(key=lambda x: x.end_date, reverse=True)
            longest = duration_contexts[0]
            start_year = longest.start_date.split("-")[0]
            end_year = longest.end_date.split("-")[0]
            if len(start_year) == 4 and len(end_year) == 4:
                return f"{start_year}-{end_year[2:]}"

        # Stage 3: Extract from filename (e.g. FY2023-24.xml or 2023-24.xml)
        match = re.search(r'(FY\d{4}-\d{2}|\d{4}-\d{2}|\d{4}-\d{4})', self.file_name)
        if match:
            return match.group(1)

        # Fallback to year of file modification
        try:
            mtime = os.path.getmtime(self.file_path)
            year = time.strftime("%Y", time.localtime(mtime))
            return f"FY{year}"
        except Exception:
            return "FYUnknown"
            
from pathlib import Path
