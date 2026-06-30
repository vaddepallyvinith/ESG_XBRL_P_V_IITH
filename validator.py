import models
from utils import logger
from typing import List, Tuple

class BRSRDataValidator:
    def __init__(self):
        self.validation_errors = 0
        self.validation_warnings = 0

    def validate_fact(self, fact: models.Fact) -> Tuple[bool, List[str]]:
        """
        Validate a single fact. Returns (is_valid, list of warning/error messages).
        """
        issues = []
        is_valid = True

        # Rule 1: Missing values (Not a failure, but log if it's a critical concept)
        if fact.normalized_value is None:
            # We don't fail for missing values since financial reports often omit details
            return True, []

        # Rule 2: Type checks based on value_type
        if fact.value_type == "numeric":
            if not isinstance(fact.normalized_value, (int, float)):
                is_valid = False
                issues.append(f"Expected numeric type for concept '{fact.concept}', got {type(fact.normalized_value)}")
        
        elif fact.value_type == "percentage":
            if not isinstance(fact.normalized_value, (int, float)):
                is_valid = False
                issues.append(f"Expected numeric percentage for concept '{fact.concept}', got {type(fact.normalized_value)}")
            else:
                # Percentage range check (e.g. should not be negative)
                if fact.normalized_value < 0:
                    issues.append(f"Percentage concept '{fact.concept}' has negative value: {fact.normalized_value}")
                    self.validation_warnings += 1
                elif fact.normalized_value > 100.0:
                    issues.append(f"Percentage concept '{fact.concept}' has value > 100: {fact.normalized_value}")
                    self.validation_warnings += 1

        elif fact.value_type == "boolean":
            if not isinstance(fact.normalized_value, bool):
                is_valid = False
                issues.append(f"Expected boolean type for concept '{fact.concept}', got {type(fact.normalized_value)}")

        elif fact.value_type == "date":
            # Already validated by regex in transformer
            pass

        # Rule 3: Non-negativity checks for counts/amounts
        concept_lower = fact.concept.lower()
        if any(keyword in concept_lower for keyword in ["numberof", "amountof", "costof", "quantityof", "turnover", "spent"]):
            if isinstance(fact.normalized_value, (int, float)) and fact.normalized_value < 0:
                # Some financial adjustments can be negative, but let's log as warning
                issues.append(f"Concept '{fact.concept}' is typically non-negative, but has value: {fact.normalized_value}")
                self.validation_warnings += 1

        if not is_valid:
            self.validation_errors += 1
            logger.warning(f"Validation failure for fact {fact.concept} in file {fact.source_file}: {', '.join(issues)}")
        elif issues:
            logger.debug(f"Validation warning for fact {fact.concept} in file {fact.source_file}: {', '.join(issues)}")

        return is_valid, issues

    def get_summary(self) -> str:
        return f"Validation Summary: {self.validation_errors} errors, {self.validation_warnings} warnings logged."
