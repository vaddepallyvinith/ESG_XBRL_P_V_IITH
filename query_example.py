import duckdb
from pathlib import Path

db_path = Path(__file__).resolve().parent / "output" / "esg_xbrl.db"

print(f"Connecting to DuckDB database at {db_path}...")
conn = duckdb.connect(str(db_path))

try:
    print("\n1. Schema details:")
    print(conn.execute("DESCRIBE esg_facts").df())

    print("\n2. Total facts per company:")
    df_counts = conn.execute("""
        SELECT company_name, COUNT(*) as fact_count
        FROM esg_facts
        GROUP BY company_name
    """).df()
    print(df_counts)

    print("\n3. Fact count by ESG Category:")
    df_cat = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM esg_facts
        GROUP BY category
        ORDER BY count DESC
    """).df()
    print(df_cat)

    print("\n4. Sample Environmental facts from Tata Consultancy Services:")
    df_env = conn.execute("""
        SELECT concept, value, normalized_value, unit_ref, start_date, end_date
        FROM esg_facts
        WHERE company_name LIKE '%Tata Consultancy%'
          AND category = 'Environmental'
          AND normalized_value IS NOT NULL
        LIMIT 5
    """).df()
    print(df_env)

finally:
    conn.close()
