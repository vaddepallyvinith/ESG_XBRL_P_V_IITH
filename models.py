import dataclasses
from typing import Dict, Optional, Any, Union

@dataclasses.dataclass
class Context:
    context_id: str
    entity_identifier: str
    entity_scheme: str
    period_type: str  # 'duration' or 'instant'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    instant_date: Optional[str] = None
    dimensions: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "entity_identifier": self.entity_identifier,
            "entity_scheme": self.entity_scheme,
            "period_type": self.period_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "instant_date": self.instant_date,
            "dimensions": self.dimensions
        }

@dataclasses.dataclass
class Unit:
    unit_id: str
    measure: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "measure": self.measure
        }

@dataclasses.dataclass
class Fact:
    concept: str
    namespace: str
    value: str
    context_ref: str
    unit_ref: Optional[str] = None
    decimals: Optional[str] = None
    normalized_value: Optional[Union[float, int, bool, str]] = None
    value_type: Optional[str] = None  # 'numeric', 'percentage', 'boolean', 'date', 'text'
    category: str = "Other"  # 'Environmental', 'Social', 'Governance', 'Other'
    company_name: Optional[str] = None
    report_year: Optional[str] = None
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "namespace": self.namespace,
            "value": self.value,
            "context_ref": self.context_ref,
            "unit_ref": self.unit_ref,
            "decimals": self.decimals,
            "normalized_value": self.normalized_value,
            "value_type": self.value_type,
            "category": self.category,
            "company_name": self.company_name,
            "report_year": self.report_year,
            "source_file": self.source_file
        }

@dataclasses.dataclass
class ReportMetadata:
    company_name: str
    report_year: str
    namespace: str
    source_file: str
    context_count: int
    unit_count: int
    fact_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "report_year": self.report_year,
            "namespace": self.namespace,
            "source_file": self.source_file,
            "context_count": self.context_count,
            "unit_count": self.unit_count,
            "fact_count": self.fact_count
        }
