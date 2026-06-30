import sys
import json
from pathlib import Path
from ontology import ESGOntologyManager
from graph_validation import KGValidator
from utils import logger

def main():
    logger.info("==================================================")
    logger.info("       ESG KNOWLEDGE GRAPH BUILD PROCESS          ")
    logger.info("==================================================")
    
    try:
        # 1. Initialize Ontology Manager
        manager = ESGOntologyManager()
        
        # 2. Build ontology metadata (schema, standard mapping, canonical concepts)
        manager.build_ontology_metadata_nodes()
        
        # 3. Build instance nodes (companies, years, facts, and relationships)
        manager.build_instance_nodes_and_relationships()
        
        # 4. Export CSVs
        manager.export_csvs()
        
        # 5. Export metadata summary
        summary = manager.generate_metadata_summary()
        
        # 6. Run Validation
        validator = KGValidator()
        report = validator.validate_all()
        
        # 7. Print results
        logger.info("\n==================================================")
        logger.info("         BUILD AND VALIDATION SUMMARY             ")
        logger.info("==================================================")
        logger.info(f"Graph Status:         {report['status']}")
        logger.info(f"Total Nodes:          {report['total_nodes']}")
        logger.info(f"Total Relationships:  {report['total_relationships']}")
        logger.info(f"Unique Node IDs:      {report['unique_node_ids']}")
        logger.info(f"Errors Logged:        {len(report['errors'])}")
        logger.info(f"Warnings Logged:      {len(report['warnings'])}")
        
        if report['errors']:
            logger.error("\nErrors encountered:")
            for err in report['errors'][:10]:
                logger.error(f" - {err}")
            if len(report['errors']) > 10:
                logger.error(f" ... and {len(report['errors']) - 10} more errors.")
            sys.exit(1)
            
        if report['warnings']:
            logger.warning("\nWarnings logged:")
            for warn in report['warnings'][:10]:
                logger.warning(f" - {warn}")
            if len(report['warnings']) > 10:
                logger.warning(f" ... and {len(report['warnings']) - 10} more warnings.")
                
        logger.info("==================================================")
        logger.info("Graph generation and validation completed successfully!")
        
    except Exception as e:
        logger.error(f"KG build process failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
