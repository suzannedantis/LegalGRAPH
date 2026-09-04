"""
data_generator.py
==================
Generates a realistic (synthetic) legal dataset for the LegalGraph system.

Produces four entity collections:
    - CaseLaw           : court opinions / precedents
    - Judge              : judges who authored / handled opinions
    - Statute             : codified law sections
    - LegalConcept   : abstract doctrines (e.g. "Duty of Care")

And four relationship collections:
    - CITES          (CaseLaw -> CaseLaw | CaseLaw -> Statute)
    - AFFIRMED_BY    (CaseLaw -> CaseLaw)   # lower court affirmed by higher court
    - OVERRULED_BY   (CaseLaw -> CaseLaw)   # precedent overturned by later case
    - HANDLED_BY     (CaseLaw -> Judge)

Also produces a simulated "implicit user citation history" matrix used by the
collaborative-filtering component of the recommender (i.e. which past cases a
set of synthetic "researcher" users looked up / cited together -- this stands
in for real usage logs).

Output: writes JSON files into ./data/ and also returns in-memory Python
objects (lists of dicts / pandas DataFrames) so downstream modules can be
unit-tested without touching disk.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

import pandas as pd

random.seed(42)  # reproducible mock data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Domain dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Judge:
    judge_id: str
    name: str
    court: str


@dataclass
class Statute:
    statute_id: str
    title: str
    code_section: str
    summary: str


@dataclass
class LegalConcept:
    concept_id: str
    name: str
    description: str


@dataclass
class CaseLaw:
    case_id: str
    title: str
    court: str
    year: int
    summary: str
    concepts: List[str] = field(default_factory=list)  # concept_ids invoked


# --------------------------------------------------------------------------- #
# Static reference pools used to build believable synthetic content
# --------------------------------------------------------------------------- #
_JUDGES = [
    ("J001", "Hon. Ramesh Kulkarni", "Supreme Court"),
    ("J002", "Hon. Sarah Whitmore", "Court of Appeals - 9th Circuit"),
    ("J003", "Hon. Anil Deshpande", "High Court of Bombay"),
    ("J004", "Hon. Elena Vasquez", "Court of Appeals - 2nd Circuit"),
    ("J005", "Hon. Priya Nair", "Supreme Court"),
    ("J006", "Hon. Thomas Reid", "District Court - S.D.N.Y."),
    ("J007", "Hon. Meera Iyer", "High Court of Delhi"),
    ("J008", "Hon. David Okafor", "Court of Appeals - 7th Circuit"),
]

_CONCEPTS = [
    ("C001", "Duty of Care", "Legal obligation to avoid acts/omissions causing foreseeable harm to others."),
    ("C002", "Negligence", "Failure to exercise reasonable care, resulting in damage to another party."),
    ("C003", "Breach of Contract", "Failure to perform any duty or obligation specified in a binding contract."),
    ("C004", "Freedom of Speech", "Constitutional protection against government restriction of expression."),
    ("C005", "Due Process", "Requirement that legal proceedings follow established rules and principles."),
    ("C006", "Vicarious Liability", "Holding one party liable for the wrongful acts of another (e.g. employer/employee)."),
    ("C007", "Proximate Cause", "The legal cause of an injury which, in natural sequence, produced the harm."),
    ("C008", "Doctrine of Precedent", "Principle (stare decisis) that courts follow rulings established in prior similar cases."),
    ("C009", "Intellectual Property Rights", "Legal rights granted for creations of the mind (patents, copyrights, trademarks)."),
    ("C010", "Right to Privacy", "Legal protection of an individual's personal information and autonomy."),
]

_STATUTES = [
    ("S001", "Indian Contract Act, 1872 - Sec. 73", "ICA-1872-S73",
     "Compensation for loss or damage caused by breach of contract."),
    ("S002", "Indian Penal Code - Sec. 304A", "IPC-S304A",
     "Causing death by negligence, not amounting to culpable homicide."),
    ("S003", "Constitution of India - Art. 19(1)(a)", "COI-ART19-1A",
     "Guarantees freedom of speech and expression to all citizens."),
    ("S004", "Constitution of India - Art. 21", "COI-ART21",
     "Protection of life and personal liberty; no deprivation except by procedure established by law."),
    ("S005", "US Code Title 17 - Copyright Act Sec. 106", "US17-S106",
     "Exclusive rights of copyright owners over reproduction and distribution."),
    ("S006", "US Restatement (Second) of Torts Sec. 282", "REST2D-TORTS-282",
     "Defines negligence as conduct falling below the standard established for protection against harm."),
    ("S007", "Information Technology Act, 2000 - Sec. 43A", "IT-ACT-S43A",
     "Compensation for failure to protect sensitive personal data."),
    ("S008", "US Fair Use Doctrine - 17 U.S.C. Sec. 107", "US17-S107",
     "Limits exclusive copyright rights for purposes such as criticism, comment, and research."),
]

_COURTS = [
    "Supreme Court", "Court of Appeals - 9th Circuit", "High Court of Bombay",
    "Court of Appeals - 2nd Circuit", "District Court - S.D.N.Y.",
    "High Court of Delhi", "Court of Appeals - 7th Circuit",
]

_CASE_TEMPLATES = [
    ("Sharma v. State Transport Corp.", "A public bus operator was found liable for {concept} after a "
     "passenger was injured due to a delayed brake response; the court examined whether the standard "
     "of care owed by common carriers was breached under {statute}."),
    ("Whitfield v. Meridian Robotics Inc.", "An employee sued a robotics manufacturer alleging {concept} "
     "after an assembly-line malfunction caused injury; the court applied {statute} to determine "
     "manufacturer liability for foreseeable defects."),
    ("Nair v. Union of India", "Petitioner challenged a state order restricting public assembly, invoking "
     "{concept} under {statute}; the court balanced individual rights against public order concerns."),
    ("Okafor v. Bright Horizon Media", "A media company was sued for unauthorized reproduction of "
     "copyrighted material, raising questions of {concept} under {statute} and permissible fair use."),
    ("Deshpande v. Continental Insurance Ltd.", "An insurer denied a claim citing policy exclusions; the "
     "court evaluated {concept} in the context of {statute} to determine enforceability of the exclusion."),
    ("Reid v. Hartwell Data Systems", "A data breach exposed customer records; plaintiffs alleged {concept} "
     "under {statute}, arguing the company failed to implement reasonable security safeguards."),
    ("Vasquez v. Cedarbrook School District", "A student's family alleged {concept} after a school failed "
     "to supervise a known hazard; the court applied {statute} to assess institutional liability."),
    ("Iyer v. Metro Housing Board", "Tenants alleged {concept} against a housing authority for unsafe "
     "premises; the court considered {statute} in evaluating landlord obligations."),
    ("Kulkarni v. National Broadcasting Trust", "A broadcaster was accused of defamatory reporting, "
     "raising {concept} concerns balanced against {statute} protections for press freedom."),
    ("Whitmore v. Silverline Pharmaceuticals", "Plaintiffs alleged the company breached {concept} in drug "
     "trial disclosures, examined under {statute} for regulatory compliance."),
]


def _gen_case_summary(concept_name: str, statute_title: str, template: tuple) -> str:
    _, body = template
    return body.format(concept=concept_name, statute=statute_title)


def generate_judges() -> List[Dict[str, Any]]:
    """Return the fixed pool of Judge entities as plain dicts."""
    return [asdict(Judge(j_id, name, court)) for j_id, name, court in _JUDGES]


def generate_statutes() -> List[Dict[str, Any]]:
    """Return the fixed pool of Statute entities as plain dicts."""
    return [asdict(Statute(s_id, title, code, summ)) for s_id, title, code, summ in _STATUTES]


def generate_concepts() -> List[Dict[str, Any]]:
    """Return the fixed pool of LegalConcept entities as plain dicts."""
    return [asdict(LegalConcept(c_id, name, desc)) for c_id, name, desc in _CONCEPTS]


def generate_cases(n: int = 24) -> List[Dict[str, Any]]:
    """
    Generate `n` synthetic CaseLaw records, each grounded in 1-2 LegalConcepts
    and referencing a real-sounding Statute in its summary text (used later for
    both graph edges and semantic embeddings).
    """
    cases: List[Dict[str, Any]] = []
    for i in range(n):
        case_id = f"CL{i + 1:03d}"
        title_template = random.choice(_CASE_TEMPLATES)
        title = f"{title_template[0]} ({2000 + random.randint(0, 25)})"
        concept = random.choice(_CONCEPTS)
        statute = random.choice(_STATUTES)
        second_concept = random.choice([c for c in _CONCEPTS if c[0] != concept[0]])
        summary = _gen_case_summary(concept[1], statute[1], title_template)
        year = 2000 + random.randint(0, 25)
        court = random.choice(_COURTS)
        cases.append(asdict(CaseLaw(
            case_id=case_id,
            title=title,
            court=court,
            year=year,
            summary=summary,
            concepts=[concept[0], second_concept[0]],
        )))
    return cases


def generate_relationships(cases: List[Dict[str, Any]],
                            judges: List[Dict[str, Any]],
                            statutes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Build CITES, AFFIRMED_BY, OVERRULED_BY, and HANDLED_BY edges across the
    generated cases, judges, and statutes.

    Returns a dict keyed by relationship type -> list of {source, target} dicts.
    """
    rels: Dict[str, List[Dict[str, str]]] = {
        "CITES": [],
        "AFFIRMED_BY": [],
        "OVERRULED_BY": [],
        "HANDLED_BY": [],
    }

    case_ids = [c["case_id"] for c in cases]

    for case in cases:
        # Every case cites at least one statute (grounding it in codified law)
        cited_statute = random.choice(statutes)
        rels["CITES"].append({"source": case["case_id"], "target": cited_statute["statute_id"]})

        # Every case is HANDLED_BY one judge
        judge = random.choice(judges)
        rels["HANDLED_BY"].append({"source": case["case_id"], "target": judge["judge_id"]})

        # ~60% of cases cite 1-2 earlier cases (lower case_id number == earlier)
        earlier_cases = [cid for cid in case_ids if cid < case["case_id"]]
        if earlier_cases and random.random() < 0.6:
            for cited_case in random.sample(earlier_cases, k=min(len(earlier_cases), random.choice([1, 2]))):
                rels["CITES"].append({"source": case["case_id"], "target": cited_case})

        # ~15% chance a later case affirms an earlier one on appeal
        if earlier_cases and random.random() < 0.15:
            affirmed_case = random.choice(earlier_cases)
            rels["AFFIRMED_BY"].append({"source": affirmed_case, "target": case["case_id"]})

        # ~10% chance a later case overrules an earlier precedent
        if earlier_cases and random.random() < 0.10:
            overruled_case = random.choice(earlier_cases)
            rels["OVERRULED_BY"].append({"source": overruled_case, "target": case["case_id"]})

    return rels


def generate_user_citation_history(cases: List[Dict[str, Any]], n_users: int = 15) -> pd.DataFrame:
    """
    Simulate an implicit-feedback matrix representing which cases synthetic
    "researcher" users have looked up / cited in past sessions. This stands in
    for real usage logs and feeds the collaborative-filtering signal in
    recommender.py.

    Returns a long-format DataFrame with columns [user_id, case_id, weight],
    where `weight` approximates implicit engagement strength (e.g. number of
    times revisited).
    """
    rows = []
    case_ids = [c["case_id"] for c in cases]
    for u in range(n_users):
        user_id = f"U{u + 1:03d}"
        n_interactions = random.randint(3, 8)
        interacted = random.sample(case_ids, k=min(n_interactions, len(case_ids)))
        for cid in interacted:
            rows.append({"user_id": user_id, "case_id": cid, "weight": random.randint(1, 5)})
    return pd.DataFrame(rows)


def generate_all(n_cases: int = 24, n_users: int = 15, persist: bool = True) -> Dict[str, Any]:
    """
    Orchestrates full mock-data generation and optionally persists everything
    to ./data/*.json (and citation_history.csv) for reuse by other modules /
    the Streamlit app without regenerating each run.
    """
    judges = generate_judges()
    statutes = generate_statutes()
    concepts = generate_concepts()
    cases = generate_cases(n_cases)
    relationships = generate_relationships(cases, judges, statutes)
    citation_history = generate_user_citation_history(cases, n_users)

    bundle = {
        "judges": judges,
        "statutes": statutes,
        "concepts": concepts,
        "cases": cases,
        "relationships": relationships,
    }

    if persist:
        os.makedirs(DATA_DIR, exist_ok=True)
        for name, payload in bundle.items():
            with open(os.path.join(DATA_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        citation_history.to_csv(os.path.join(DATA_DIR, "citation_history.csv"), index=False)

    bundle["citation_history"] = citation_history
    return bundle


if __name__ == "__main__":
    data = generate_all()
    print(f"Generated {len(data['cases'])} cases, {len(data['judges'])} judges, "
          f"{len(data['statutes'])} statutes, {len(data['concepts'])} concepts.")
    total_edges = sum(len(v) for v in data["relationships"].values())
    print(f"Generated {total_edges} relationship edges across "
          f"{list(data['relationships'].keys())}.")
    print(f"Simulated citation history: {len(data['citation_history'])} interaction rows "
          f"across {data['citation_history']['user_id'].nunique()} users.")
    print(f"Data written to: {DATA_DIR}")
