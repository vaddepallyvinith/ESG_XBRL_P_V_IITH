import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from utils import logger
from config import OUTPUT_DIR

class KGValidator:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.nodes_csv = output_dir / "nodes.csv"
        self.relationships_csv = output_dir / "relationships.csv"
        self.metadata_json = output_dir / "ontology_metadata.json"
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation tests and return a diagnostic report."""
        logger.info("Starting ESG Knowledge Graph validation...")
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check files exist
        if not self.nodes_csv.exists():
            errors.append(f"Nodes CSV file not found at {self.nodes_csv}")
            return {"status": "FAILED", "errors": errors, "warnings": warnings}
        if not self.relationships_csv.exists():
            errors.append(f"Relationships CSV file not found at {self.relationships_csv}")
            return {"status": "FAILED", "errors": errors, "warnings": warnings}
        if not self.metadata_json.exists():
            errors.append(f"Metadata JSON file not found at {self.metadata_json}")
            return {"status": "FAILED", "errors": errors, "warnings": warnings}

        # 1. Load nodes and check for duplicate IDs and empty fields
        node_ids: Set[str] = set()
        node_types: Set[str] = set()
        total_nodes = 0
        
        with open(self.nodes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                total_nodes += 1
                nid = row.get("id:ID")
                lbl = row.get("label:LABEL")
                
                # Check for empty ID
                if not nid:
                    errors.append(f"Empty ID in nodes.csv at row {idx+1}")
                    continue
                    
                # Check for empty Label
                if not lbl:
                    errors.append(f"Empty Label for node '{nid}' in nodes.csv at row {idx+1}")
                    continue
                
                # Check for duplicate ID
                if nid in node_ids:
                    errors.append(f"Duplicate Node ID found: '{nid}' at row {idx+1}")
                else:
                    node_ids.add(nid)
                    node_types.add(lbl)

        logger.info(f"Loaded {total_nodes} nodes from {self.nodes_csv} with 0 duplicate IDs.")

        # 2. Load relationships and check referential integrity
        total_rels = 0
        valid_rel_types = {"REPORTS", "HAS_METRIC", "BELONGS_TO", "MAPS_TO", "DISCLOSED_IN"}
        
        with open(self.relationships_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                total_rels += 1
                start_id = row.get("start_id:START_ID")
                end_id = row.get("end_id:END_ID")
                rel_type = row.get("type:TYPE")
                
                # Check empty values
                if not start_id or not end_id:
                    errors.append(f"Empty START_ID or END_ID in relationships.csv at row {idx+1}")
                    continue
                if not rel_type:
                    errors.append(f"Empty TYPE in relationships.csv at row {idx+1}")
                    continue
                    
                # Check referential integrity
                if start_id not in node_ids:
                    errors.append(f"Referential Integrity Violation: START_ID '{start_id}' does not exist in nodes.csv (row {idx+1})")
                if end_id not in node_ids:
                    errors.append(f"Referential Integrity Violation: END_ID '{end_id}' does not exist in nodes.csv (row {idx+1})")
                    
                # Check valid relationship type
                if rel_type not in valid_rel_types:
                    warnings.append(f"Unexpected relationship type: '{rel_type}' at row {idx+1}")

        logger.info(f"Loaded {total_rels} relationships from {self.relationships_csv} and validated referential integrity.")

        # 3. Check Canonical Concepts coverage
        expected_canonical = {
            "Scope1Emission", "Scope2Emission", "RenewableEnergy", "WaterWithdrawal",
            "WasteGenerated", "WomenEmployees", "BoardIndependence", "CSRSpending"
        }
        for concept in expected_canonical:
            if concept not in node_ids:
                errors.append(f"Schema Coverage Violation: Canonical concept '{concept}' is missing from the exported nodes.")

        # 4. Check Metadata alignment
        try:
            with open(self.metadata_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("nodes_count") != total_nodes:
                warnings.append(f"Metadata node count ({metadata.get('nodes_count')}) does not match nodes.csv count ({total_nodes})")
            if metadata.get("relationships_count") != total_rels:
                warnings.append(f"Metadata relationship count ({metadata.get('relationships_count')}) does not match relationships.csv count ({total_rels})")
        except Exception as e:
            errors.append(f"Failed to validate metadata alignment: {e}")

        # Final diagnostic report
        status = "PASSED" if not errors else "FAILED"
        report = {
            "status": status,
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "unique_node_ids": len(node_ids),
            "errors": errors,
            "warnings": warnings
        }
        
        if status == "PASSED":
            logger.info("ESG Knowledge Graph validation PASSED successfully.")
        else:
            logger.error(f"ESG Knowledge Graph validation FAILED with {len(errors)} errors and {len(warnings)} warnings.")
            
        return report
