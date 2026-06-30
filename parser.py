import os
from typing import Dict, List, Tuple, Optional, Any
from bs4 import BeautifulSoup
import models
from utils import logger

# Try importing lxml, keep fallback ready
try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    logger.warning("lxml is not available. Falling back to BeautifulSoup.")

class XBRLParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.namespaces: Dict[str, str] = {}
        self.contexts: Dict[str, models.Context] = {}
        self.units: Dict[str, models.Unit] = {}
        self.facts: List[models.Fact] = []

    def parse(self) -> Tuple[Dict[str, models.Context], Dict[str, models.Unit], List[models.Fact], Dict[str, str]]:
        """Parse the XBRL file and extract contexts, units, and facts."""
        try:
            if LXML_AVAILABLE:
                self._parse_with_lxml()
            else:
                self._parse_with_bs4()
        except Exception as e:
            logger.error(f"Error parsing {self.file_path} with lxml: {e}. Trying BeautifulSoup fallback...")
            try:
                self._parse_with_bs4()
            except Exception as bs_err:
                logger.error(f"BS4 fallback also failed for {self.file_path}: {bs_err}")
                raise bs_err
                
        return self.contexts, self.units, self.facts, self.namespaces

    def _parse_with_lxml(self):
        """Parse using lxml for speed and namespace handling."""
        # Parse XML tree
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        tree = etree.parse(self.file_path, parser=parser)
        root = tree.getroot()
        
        # Get namespaces
        self.namespaces = {k if k is not None else "default": v for k, v in root.nsmap.items()}
        
        # Define XPath namespaces helper
        ns = {
            "xbrli": "http://www.xbrl.org/2003/instance",
            "xbrldi": "http://xbrl.org/2006/xbrldi"
        }
        # Update with document's actual nsmap to avoid missing prefixes
        for prefix, uri in root.nsmap.items():
            if prefix:
                ns[prefix] = uri
                
        # Helper function to get tag name without namespace
        def get_local_name(qname: str) -> str:
            if "}" in qname:
                return qname.split("}")[1]
            return qname
            
        def get_namespace_uri(qname: str) -> str:
            if "}" in qname:
                return qname.split("}")[0].strip("{")
            return ""

        # 1. Parse Contexts
        # Search for context tags (either with prefix xbrli or default)
        context_nodes = root.findall(".//{http://www.xbrl.org/2003/instance}context")
        # If none, try without namespace namespace-agnostic search
        if not context_nodes:
            context_nodes = root.xpath("//*[local-name()='context']")

        for ctx in context_nodes:
            ctx_id = ctx.get("id")
            if not ctx_id:
                continue
                
            # Entity Identifier
            identifier_node = ctx.find(".//{http://www.xbrl.org/2003/instance}identifier")
            if identifier_node is None:
                identifier_node = ctx.xpath(".//*[local-name()='identifier']")
                identifier_node = identifier_node[0] if identifier_node else None
                
            ent_id = ""
            ent_scheme = ""
            if identifier_node is not None:
                ent_id = identifier_node.text or ""
                ent_scheme = identifier_node.get("scheme") or ""

            # Period
            period_node = ctx.find(".//{http://www.xbrl.org/2003/instance}period")
            if period_node is None:
                period_node = ctx.xpath(".//*[local-name()='period']")
                period_node = period_node[0] if period_node else None
                
            period_type = "duration"
            start_date = None
            end_date = None
            instant_date = None
            
            if period_node is not None:
                # Instant
                inst_node = period_node.find(".//{http://www.xbrl.org/2003/instance}instant")
                if inst_node is None:
                    inst_node = period_node.xpath(".//*[local-name()='instant']")
                    inst_node = inst_node[0] if inst_node else None
                    
                if inst_node is not None:
                    period_type = "instant"
                    instant_date = inst_node.text
                else:
                    start_node = period_node.find(".//{http://www.xbrl.org/2003/instance}startDate")
                    if start_node is None:
                        start_node = period_node.xpath(".//*[local-name()='startDate']")
                        start_node = start_node[0] if start_node else None
                        
                    end_node = period_node.find(".//{http://www.xbrl.org/2003/instance}endDate")
                    if end_node is None:
                        end_node = period_node.xpath(".//*[local-name()='endDate']")
                        end_node = end_node[0] if end_node else None
                        
                    if start_node is not None and end_node is not None:
                        period_type = "duration"
                        start_date = start_node.text
                        end_date = end_node.text

            # Scenarios/Dimensions
            dimensions = {}
            scenario_node = ctx.find(".//{http://www.xbrl.org/2003/instance}scenario")
            if scenario_node is None:
                scenario_node = ctx.xpath(".//*[local-name()='scenario']")
                scenario_node = scenario_node[0] if scenario_node else None
                
            if scenario_node is not None:
                # Explicit members
                explicit_members = scenario_node.findall(".//{http://xbrl.org/2006/xbrldi}explicitMember")
                if not explicit_members:
                    explicit_members = scenario_node.xpath(".//*[local-name()='explicitMember']")
                for mem in explicit_members:
                    dim = mem.get("dimension")
                    val = mem.text
                    if dim and val:
                        dimensions[dim] = val
                        
                # Typed members
                typed_members = scenario_node.findall(".//{http://xbrl.org/2006/xbrldi}typedMember")
                if not typed_members:
                    typed_members = scenario_node.xpath(".//*[local-name()='typedMember']")
                for mem in typed_members:
                    dim = mem.get("dimension")
                    # Typed member has nested child containing value
                    if dim:
                        children = list(mem)
                        if children:
                            # Use child tag name + text
                            child = children[0]
                            val = child.text or ""
                            dimensions[dim] = val
                        else:
                            dimensions[dim] = mem.text or ""

            self.contexts[ctx_id] = models.Context(
                context_id=ctx_id,
                entity_identifier=ent_id,
                entity_scheme=ent_scheme,
                period_type=period_type,
                start_date=start_date,
                end_date=end_date,
                instant_date=instant_date,
                dimensions=dimensions
            )

        # 2. Parse Units
        unit_nodes = root.findall(".//{http://www.xbrl.org/2003/instance}unit")
        if not unit_nodes:
            unit_nodes = root.xpath("//*[local-name()='unit']")
            
        for unit in unit_nodes:
            unit_id = unit.get("id")
            if not unit_id:
                continue
            
            measure_node = unit.find(".//{http://www.xbrl.org/2003/instance}measure")
            if measure_node is None:
                measure_node = unit.xpath(".//*[local-name()='measure']")
                measure_node = measure_node[0] if measure_node else None
                
            measure = ""
            if measure_node is not None:
                measure = measure_node.text or ""
                
            self.units[unit_id] = models.Unit(unit_id=unit_id, measure=measure)

        # 3. Parse Facts
        # Facts are nodes that have contextRef attribute
        fact_nodes = root.xpath("//*[@contextRef]")
        for fn in fact_nodes:
            concept = get_local_name(fn.tag)
            ns_uri = get_namespace_uri(fn.tag)
            
            # Skip core XBRL structure tags that might have contextRef (unlikely, but safe)
            if concept in ["context", "unit", "xbrl"]:
                continue
                
            self.facts.append(models.Fact(
                concept=concept,
                namespace=ns_uri,
                value=fn.text or "",
                context_ref=fn.get("contextRef"),
                unit_ref=fn.get("unitRef"),
                decimals=fn.get("decimals"),
                source_file=os.path.basename(self.file_path)
            ))

    def _parse_with_bs4(self):
        """Parse using BeautifulSoup as a robust fallback."""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        soup = BeautifulSoup(content, "xml")
        
        # 1. Parse Contexts
        contexts = soup.find_all(["context", "xbrli:context"])
        for ctx in contexts:
            ctx_id = ctx.get("id")
            if not ctx_id:
                continue
                
            # Entity identifier
            ident = ctx.find(["identifier", "xbrli:identifier"])
            ent_id = ident.text.strip() if ident else ""
            ent_scheme = ident.get("scheme", "") if ident else ""
            
            # Period
            period = ctx.find(["period", "xbrli:period"])
            period_type = "duration"
            start_date = None
            end_date = None
            instant_date = None
            
            if period:
                inst = period.find(["instant", "xbrli:instant"])
                if inst:
                    period_type = "instant"
                    instant_date = inst.text.strip()
                else:
                    sd = period.find(["startDate", "xbrli:startDate"])
                    ed = period.find(["endDate", "xbrli:endDate"])
                    if sd and ed:
                        period_type = "duration"
                        start_date = sd.text.strip()
                        end_date = ed.text.strip()
                        
            # Dimensions
            dimensions = {}
            scenario = ctx.find(["scenario", "xbrli:scenario"])
            if scenario:
                explicit_mems = scenario.find_all(["explicitMember", "xbrldi:explicitMember"])
                for mem in explicit_mems:
                    dim = mem.get("dimension")
                    val = mem.text.strip()
                    if dim and val:
                        dimensions[dim] = val
                        
                typed_mems = scenario.find_all(["typedMember", "xbrldi:typedMember"])
                for mem in typed_mems:
                    dim = mem.get("dimension")
                    if dim:
                        # Find first child
                        children = [c for c in mem.children if c.name]
                        if children:
                            dimensions[dim] = children[0].text.strip()
                        else:
                            dimensions[dim] = mem.text.strip()

            self.contexts[ctx_id] = models.Context(
                context_id=ctx_id,
                entity_identifier=ent_id,
                entity_scheme=ent_scheme,
                period_type=period_type,
                start_date=start_date,
                end_date=end_date,
                instant_date=instant_date,
                dimensions=dimensions
            )

        # 2. Parse Units
        units = soup.find_all(["unit", "xbrli:unit"])
        for unit in units:
            unit_id = unit.get("id")
            if not unit_id:
                continue
            meas = unit.find(["measure", "xbrli:measure"])
            measure = meas.text.strip() if meas else ""
            self.units[unit_id] = models.Unit(unit_id=unit_id, measure=measure)

        # 3. Parse Facts
        # Any element that has a contextRef attribute
        all_elements = soup.find_all(True)
        for el in all_elements:
            if el.has_attr("contextRef"):
                # Tag names in BS4 might be prefixed (e.g. "in-capmkt:NameOfListedEntity") or local
                tag_name = el.name
                concept = tag_name.split(":")[-1] if ":" in tag_name else tag_name
                
                # BS4 does not easily give namespace URI directly for a tag without scanning parents,
                # but we can look it up in prefix namespaces or set default
                ns_prefix = tag_name.split(":")[0] if ":" in tag_name else ""
                ns_uri = "" # Will resolve dynamically or fallback
                
                self.facts.append(models.Fact(
                    concept=concept,
                    namespace=ns_uri,
                    value=el.text.strip(),
                    context_ref=el.get("contextRef"),
                    unit_ref=el.get("unitRef"),
                    decimals=el.get("decimals"),
                    source_file=os.path.basename(self.file_path)
                ))
