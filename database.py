import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import duckdb
import models
from config import PARQUET_DIR, DB_PATH, FACTS_TABLE, REPORTS_TABLE
from utils import logger

class ESGDatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.facts_table = FACTS_TABLE
        self.reports_table = REPORTS_TABLE
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB tables if they do not exist."""
        # Ensure parent folder exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = duckdb.connect(str(self.db_path))
        try:
            # Create facts table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {FACTS_TABLE} (
                    concept VARCHAR,
                    namespace VARCHAR,
                    value TEXT,
                    context_ref VARCHAR,
                    unit_ref VARCHAR,
                    decimals VARCHAR,
                    normalized_value VARCHAR,
                    value_type VARCHAR,
                    category VARCHAR,
                    company_name VARCHAR,
                    report_year VARCHAR,
                    source_file VARCHAR,
                    period_type VARCHAR,
                    start_date VARCHAR,
                    end_date VARCHAR,
                    instant_date VARCHAR,
                    dimensions_json VARCHAR
                )
            """)
            
            # Create metadata table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {REPORTS_TABLE} (
                    company_name VARCHAR,
                    report_year VARCHAR,
                    namespace VARCHAR,
                    source_file VARCHAR,
                    context_count INTEGER,
                    unit_count INTEGER,
                    fact_count INTEGER,
                    ingested_at TIMESTAMP
                )
            """)
            logger.info(f"Initialized DuckDB database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            raise e
        finally:
            conn.close()

    def persist_batch(
        self,
        facts: List[models.Fact],
        contexts: Dict[str, models.Context],
        metadata: models.ReportMetadata
    ):
        """Idempotently persist facts and metadata to Parquet and DuckDB."""
        if not facts:
            logger.warning(f"No facts to persist for {metadata.company_name} ({metadata.report_year})")
            return

        # Prepare enriched facts list
        flat_facts = []
        for fact in facts:
            fact_dict = fact.to_dict()
            
            # Enriched fields from context
            ctx = contexts.get(fact.context_ref)
            if ctx:
                fact_dict["period_type"] = ctx.period_type
                fact_dict["start_date"] = ctx.start_date
                fact_dict["end_date"] = ctx.end_date
                fact_dict["instant_date"] = ctx.instant_date
                fact_dict["dimensions_json"] = json.dumps(ctx.dimensions)
            else:
                fact_dict["period_type"] = None
                fact_dict["start_date"] = None
                fact_dict["end_date"] = None
                fact_dict["instant_date"] = None
                fact_dict["dimensions_json"] = json.dumps({})
                
            # Cast normalized value to string for uniform storage in DuckDB
            if fact_dict["normalized_value"] is not None:
                fact_dict["normalized_value"] = str(fact_dict["normalized_value"])
                
            flat_facts.append(fact_dict)

        # Convert to Pandas DataFrame
        df = pd.DataFrame(flat_facts)

        # 1. Save to Partitioned Parquet
        # Partition columns: company_name, report_year
        try:
            logger.info(f"Writing partitioned Parquet for {metadata.company_name} - {metadata.report_year}")
            # Ensure the directories are clean if we overwrite or let pandas handle it
            # pandas.to_parquet handles appending/partitioning, but let's write to a temp file
            # or directly partition it. We specify engine='pyarrow'
            df.to_parquet(
                str(PARQUET_DIR),
                engine='pyarrow',
                partition_cols=['company_name', 'report_year'],
                index=False
            )
        except Exception as e:
            logger.error(f"Error writing Parquet: {e}")

        # 2. Save to DuckDB
        conn = duckdb.connect(str(self.db_path))
        try:
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Delete existing records to maintain idempotency (prevent duplicates on rerun)
            conn.execute(
                f"DELETE FROM {FACTS_TABLE} WHERE company_name = ? AND report_year = ?",
                (metadata.company_name, metadata.report_year)
            )
            conn.execute(
                f"DELETE FROM {REPORTS_TABLE} WHERE company_name = ? AND report_year = ?",
                (metadata.company_name, metadata.report_year)
            )
            
            # Insert facts using pandas integration
            conn.execute(f"INSERT INTO {FACTS_TABLE} SELECT * FROM df")
            
            # Insert report metadata
            now = datetime.datetime.now()
            conn.execute(
                f"INSERT INTO {REPORTS_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    metadata.company_name,
                    metadata.report_year,
                    metadata.namespace,
                    metadata.source_file,
                    metadata.context_count,
                    metadata.unit_count,
                    metadata.fact_count,
                    now
                )
            )
            
            conn.execute("COMMIT")
            logger.info(f"Successfully committed {len(facts)} facts to DuckDB for {metadata.company_name} ({metadata.report_year})")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to persist facts to DuckDB: {e}")
            raise e
        finally:
            conn.close()

    def get_ingestion_summary(self) -> pd.DataFrame:
        """Retrieve a summary of all ingested files."""
        conn = duckdb.connect(str(self.db_path))
        try:
            df = conn.execute(f"SELECT * FROM {REPORTS_TABLE}").df()
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve summary from DuckDB: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
