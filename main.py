import argparse
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional
import duckdb

import models
from config import TCS_XBRL_DIR, RELIANCE_XBRL_DIR
from parser import XBRLParser
from extractor import BRSRDataExtractor
from transformer import BRSRDataTransformer
from validator import BRSRDataValidator
from database import ESGDatabaseManager
from utils import logger, scan_xbrl_files, ResumeState, run_concurrently, time_it

# Threading lock for database writes (DuckDB and Parquet)
db_lock = threading.Lock()

class ESGPipeline:
    def __init__(self, force_run: bool = False, max_workers: int = 4):
        self.force_run = force_run
        self.max_workers = max_workers
        
        # Initialize pipeline components
        self.resume_state = ResumeState()
        if self.force_run:
            logger.info("Force flag set. Clearing previous resume state.")
            self.resume_state.clear()
            self.resume_state = ResumeState() # Re-init fresh
            
        self.db_manager = ESGDatabaseManager()
        self.transformer = BRSRDataTransformer()
        self.validator = BRSRDataValidator()

    def process_single_file(self, file_path: Path) -> bool:
        """Process a single XBRL file: parse, extract, transform, validate, and persist."""
        file_name = file_path.name
        logger.info(f"--- Starting processing of {file_name} ---")
        
        try:
            # 1. Parse XBRL
            parser = XBRLParser(str(file_path.resolve()))
            contexts, units, facts, namespaces = parser.parse()
            
            # 2. Extract metadata and bind
            extractor = BRSRDataExtractor(str(file_path.resolve()))
            metadata, enriched_facts = extractor.extract_metadata_and_bind(
                contexts, units, facts, namespaces
            )
            
            # 3. Transform (Normalize & Categorize)
            processed_facts = []
            for fact in enriched_facts:
                normalized_fact = self.transformer.normalize_and_categorize(fact)
                
                # 4. Validate
                self.validator.validate_fact(normalized_fact)
                processed_facts.append(normalized_fact)
            
            # 5. Persist (Thread-safe write)
            with db_lock:
                self.db_manager.persist_batch(processed_facts, contexts, metadata)
                
            # 6. Mark as processed
            self.resume_state.mark_processed(
                file_path,
                metadata={
                    "company_name": metadata.company_name,
                    "report_year": metadata.report_year,
                    "fact_count": metadata.fact_count
                }
            )
            logger.info(f"--- Finished processing of {file_name} successfully ---")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}", exc_info=True)
            return False

    @time_it
    def run(self):
        """Orchestrate the ingestion run."""
        logger.info("Initializing ESG XBRL Ingestion Pipeline...")
        
        # Scan files
        target_dirs = [TCS_XBRL_DIR, RELIANCE_XBRL_DIR]
        logger.info(f"Scanning target directories: {[str(d) for d in target_dirs]}")
        all_files = scan_xbrl_files(target_dirs)
        
        if not all_files:
            logger.warning("No XBRL files discovered. Pipeline exiting.")
            return

        # Filter processed files
        files_to_process = []
        skipped_count = 0
        for f in all_files:
            if not self.force_run and self.resume_state.is_processed(f):
                skipped_count += 1
                logger.debug(f"Skipping already processed file: {f.name}")
            else:
                files_to_process.append(f)
                
        logger.info(f"Total files discovered: {len(all_files)}")
        logger.info(f"Skipped (already processed): {skipped_count}")
        logger.info(f"Files to process in this run: {len(files_to_process)}")
        
        if not files_to_process:
            logger.info("All files are up to date. Nothing to process.")
            self._print_overall_summary(0, skipped_count, 0)
            return

        # Process concurrently
        logger.info(f"Starting parallel processing with {self.max_workers} workers...")
        success_flags = run_concurrently(self.process_single_file, files_to_process, max_workers=self.max_workers)
        
        success_count = sum(1 for x in success_flags if x)
        failure_count = len(files_to_process) - success_count
        
        self._print_overall_summary(success_count, skipped_count, failure_count)

    def _print_overall_summary(self, success: int, skipped: int, failed: int):
        """Print overall ingestion report and database statistics."""
        summary = self.db_manager.get_ingestion_summary()
        
        report = []
        report.append("\n" + "="*50)
        report.append("          ESG XBRL PIPELINE RUN SUMMARY")
        report.append("="*50)
        report.append(f"Files Processed: {success}")
        report.append(f"Files Skipped:   {skipped}")
        report.append(f"Files Failed:    {failed}")
        report.append(self.validator.get_summary())
        
        if not summary.empty:
            report.append("\nDatabase Ingestion Details:")
            report.append(summary.to_string(index=False))
            
            # Fetch some category breakdown
            conn = duckdb.connect(str(self.db_manager.db_path))
            try:
                cat_df = conn.execute(f"SELECT category, COUNT(*) as count FROM {self.db_manager.facts_table} GROUP BY category").df()
                report.append("\nFact Categorization Breakdown:")
                report.append(cat_df.to_string(index=False))
            except Exception as e:
                logger.error(f"Could not fetch fact breakdown: {e}")
            finally:
                conn.close()
                
        report.append("="*50)
        
        logger.info("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="ESG BRSR XBRL Ingestion Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all files (clears resume state)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)"
    )
    
    args = parser.parse_args()
    
    pipeline = ESGPipeline(force_run=args.force, max_workers=args.workers)
    pipeline.run()

if __name__ == "__main__":
    main()
