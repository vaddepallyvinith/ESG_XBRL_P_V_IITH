import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent folder .env if it exists
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "financial_dataset"

# Input paths
TCS_XBRL_DIR = DATASET_DIR / "TCS" / "XBRL"
RELIANCE_XBRL_DIR = DATASET_DIR / "Reliance" / "XBRL"

# Output paths
OUTPUT_DIR = BASE_DIR / "output"
PARQUET_DIR = OUTPUT_DIR / "parquet"
DB_PATH = OUTPUT_DIR / "esg_xbrl.db"
RESUME_STATE_PATH = OUTPUT_DIR / "resume_state.json"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Database table names
FACTS_TABLE = "esg_facts"
REPORTS_TABLE = "reports_metadata"

# Namespace mapping for XBRL parsing
NAMESPACES = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "xbrldi": "http://xbrl.org/2006/xbrldi",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "in-capmkt": "https://www.sebi.gov.in/xbrl/2024-04-30/in-capmkt",
    "in-capmkt-types": "https://www.sebi.gov.in/xbrl/2024-04-30/in-capmkt-types",
}

# ESG Taxonomy keyword lists for concept categorization (case-insensitive checks)
ENVIRONMENTAL_KEYWORDS = [
    "energy", "electricity", "fuel", "coal", "diesel", "gas", "renewable", "solar", "wind", "hydro", "biomass",
    "water", "withdrawal", "discharge", "consumption", "recycle", "reused", "harvesting", "wastewater",
    "emission", "greenhouse", "ghg", "scope1", "scope2", "scope3", "carbon", "co2", "pollutant", "nox", "sox",
    "waste", "hazardous", "plastic", "ewaste", "effluent", "spill", "landfill", "disposal",
    "environmental", "ecology", "biodiversity", "habitat", "forest", "tree", "plantation", "nature",
    "rawmaterial", "resource", "circular", "sustainablesourcing", "lifecycle", "spills"
]

SOCIAL_KEYWORDS = [
    "employee", "worker", "staff", "personnel", "fte", "contractor", "temporary", "permanent",
    "gender", "male", "female", "othergender", "diversity", "inclusion", "differentlyabled", "disabled", "special",
    "wage", "salary", "remuneration", "minimumwage", "equalopportunity", "turnover", "attrition",
    "health", "safety", "accident", "injury", "fatality", "medical", "insurance", "maternity", "paternity", "daycare",
    "training", "skill", "development", "education", "upgradation", "benefits", "welfare",
    "humanright", "childlabour", "forcedlabour", "discrimination", "sexualharassment", "posh", "grievance", "complaint",
    "csr", "corporatesocialresponsibility", "community", "localcommunity", "socialimpact",
    "consumer", "customer", "productsafety", "recall", "dataprivacy", "cybersecurity", "databreach", "pii",
    "vulnerable", "marginalized", "tribal", "indigenous", "collectivebargaining", "tradeunion", "association"
]

GOVERNANCE_KEYWORDS = [
    "board", "director", "chairman", "ceo", "cfo", "kmp", "committee", "independentdirector", "executivedirector",
    "governance", "corporategovernance", "policy", "codeofconduct", "ethics", "anticorruption", "bribery", "whistleblower", "vigilmechanism",
    "shareholder", "stakeholder", "investor", "annualgeneralmeeting", "agm", "voting", "resolution",
    "compliance", "fine", "penalty", "lawsuit", "legalproceedings", "regulatoryaction", "showcause", "prosecution",
    "audit", "auditor", "internalaudit", "statutoryauditor", "assurance", "verification",
    "relatedparty", "transaction", "conflictofinterest", "remunerationofdirector"
]

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Falls back to gemini-1.5-flash or gemini-2.5-flash as the standard model
GEMINI_MODEL = "gemini-1.5-flash"
