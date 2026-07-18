import logging
from typing import List, Dict, Tuple
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

import logging
from typing import List, Dict, Tuple
from matcher.models import OntologyConcept

logger = logging.getLogger(__name__)

class PropertyMatcher:
    """AML-inspired Property Matcher.
    Compares data types, units of measure, and dimension compatibility.
    """
    def __init__(self, config: dict):
        self.config = config.get("property_matcher", {})
        
        # Unit dimension categories for conversion compatibility
        self.UNIT_CATEGORIES = {
            "volume": ["kilolitre", "kl", "litre", "liter", "l", "megalitre", "ml", "cubic meter", "m3", "m³", "kilolitres", "litres", "megalitres"],
            "energy": ["joule", "joules", "megajoule", "megajoules", "mj", "gigajoule", "gigajoules", "gj", "kilowatt hour", "kwh", "megawatt hour", "mwh", "wh"],
            "emissions": ["tco2e", "co2e", "tonnes of co2", "tonnes co2", "tonnes", "t", "kg co2", "metric tonnes", "tonne", "kg", "kilograms"],
            "waste": ["tonnes", "t", "kg", "grams", "metric tonnes", "tonne", "kilograms"],
            "percentage": ["percent", "%", "percentage", "proportion", "fraction"],
            "currency": ["rupee", "rupees", "inr", "usd", "dollar", "dollars", "currency"]
        }

    def _get_unit_category(self, unit: str) -> str:
        if not unit:
            return "unknown"
        u_lower = unit.lower()
        for cat, synonyms in self.UNIT_CATEGORIES.items():
            if any(syn in u_lower for syn in synonyms):
                return cat
        return "unknown"

    def match(self, source_concepts: List[OntologyConcept], target_concepts: List[OntologyConcept]) -> Dict[Tuple[str, str], float]:
        logger.info("Running Property Matcher...")
        scores = {}
        
        for s_concept in source_concepts:
            s_unit_cat = self._get_unit_category(s_concept.unit)
            s_is_intensity = any(w in (s_concept.unit or "").lower() for w in ["per", "intensity", "/", "turnover", "revenue"])
            
            for t_concept in target_concepts:
                t_unit_cat = self._get_unit_category(t_concept.unit)
                t_is_intensity = any(w in (t_concept.unit or "").lower() for w in ["per", "intensity", "/", "turnover", "revenue"])
                
                # 1. Compare Datatypes
                datatype_compat = 0.5
                if s_concept.datatype and t_concept.datatype:
                    s_dt = s_concept.datatype.lower()
                    t_dt = t_concept.datatype.lower()
                    
                    s_is_num = any(w in s_dt for w in ["int", "decimal", "float", "double", "number"])
                    t_is_num = any(w in t_dt for w in ["int", "decimal", "float", "double", "number"])
                    
                    if s_dt == t_dt:
                        datatype_compat = 1.0
                    elif s_is_num and t_is_num:
                        datatype_compat = 1.0
                    elif s_is_num or t_is_num:
                        # Numeric vs non-numeric (e.g. text/string)
                        datatype_compat = 0.5
                    else:
                        datatype_compat = 0.8  # both non-numeric string/text
                
                # 2. Compare Units
                unit_compat = 0.5
                if s_concept.unit and t_concept.unit:
                    s_ut = s_concept.unit.lower().strip()
                    t_ut = t_concept.unit.lower().strip()
                    
                    if s_ut == t_ut:
                        unit_compat = 1.0
                    elif s_unit_cat != "unknown" and s_unit_cat == t_unit_cat:
                        # Same dimension category (convertible volume, mass, energy etc)
                        if s_is_intensity == t_is_intensity:
                            unit_compat = 0.95
                        else:
                            # One is intensity ratio, other is absolute amount
                            unit_compat = 0.70
                    elif s_is_intensity and t_is_intensity:
                        unit_compat = 0.60
                    else:
                        unit_compat = 0.20
                        
                scores[(s_concept.uri, t_concept.uri)] = float((datatype_compat * 0.5) + (unit_compat * 0.5))
                
        return scores
