IPO_REGISTRY = {
    "shadowfax": ["data/raw/shadowfax/Shadowfax_DRHP.pdf"],
    "tata_capital": ["data/raw/tata_capital/TataCapital_DRHP.pdf"]
}

def get_documents_for_company(company: str):
    return IPO_REGISTRY.get(company.lower(), [])
