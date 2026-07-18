import logging
from typing import List
from matcher.models import FinalMapping

logger = logging.getLogger(__name__)

import logging
from typing import List
from matcher.models import FinalMapping

logger = logging.getLogger(__name__)

class SKOSMapper:
    """SKOS Alignment Mapper.
    Maps verified alignments to standard SKOS mapping properties.
    """
    def __init__(self, config: dict):
        self.config = config.get("skos_mapper", {})

    def map_to_skos(self, mappings: List[FinalMapping]) -> List[FinalMapping]:
        logger.info("Mapping alignments to SKOS standard properties...")
        
        for m in mappings:
            rel = m.relationship.lower().strip()
            if rel == "equivalent":
                m.ontology_path = "http://www.w3.org/2004/02/skos/core#exactMatch"
            elif rel == "partial equivalent" or rel == "partial":
                m.ontology_path = "http://www.w3.org/2004/02/skos/core#closeMatch"
            elif rel == "broader":
                m.ontology_path = "http://www.w3.org/2004/02/skos/core#broadMatch"
            elif rel == "narrower":
                m.ontology_path = "http://www.w3.org/2004/02/skos/core#narrowMatch"
            else:
                m.ontology_path = "http://www.w3.org/2004/02/skos/core#relatedMatch"
                
        return mappings
