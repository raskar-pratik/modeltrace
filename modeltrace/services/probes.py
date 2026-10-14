"""Built-in original screening probes and JSON/JSONL import validation."""

import json
from typing import Any, Dict, List

from modeltrace.schemas import ProbePromptItem

VALIDATION_LABEL = "UNVALIDATED_FOR_IDENTITY_DISCRIMINATION"

BUILTIN_PROBES: List[Dict[str, Any]] = [
    # Family 1: Extraction (5 prompts)
    {
        "prompt_id": "ext-01",
        "family": "extraction",
        "title": "Invoice Line-Item Extraction",
        "prompt_text": (
            "Extract the vendor name, invoice date (YYYY-MM-DD), and total amount in USD from "
            "this memo: 'On 14 March 2025, Northwind Optics billed us $1,420.50 for lens coatings "
            "and $179.50 for shipping.' Reply in one compact sentence."
        ),
        "expected_constraint": "Vendor: Northwind Optics; Date: 2025-03-14; Total: 1600.00",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "ext-02",
        "family": "extraction",
        "title": "ClinicalVital Sign Tuple Extraction",
        "prompt_text": (
            "From the note: 'Pt arrived at 08:15 with HR 78 bpm, BP 118/76 mmHg, SpO2 98% on room air, "
            "temp 36.8C', list only the four numerical vital measurements separated by semicolons."
        ),
        "expected_constraint": "78 bpm; 118/76 mmHg; 98%; 36.8C",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "ext-03",
        "family": "extraction",
        "title": "Changelog Breaking Change Identification",
        "prompt_text": (
            "Read the release notes: 'v2.4.0: (1) Added dark theme toggle; (2) Renamed header "
            "X-Client-Token to X-Auth-Token and removed legacy header support; (3) Reduced memory "
            "footprint by 12%.' State only the breaking change in under 20 words."
        ),
        "expected_constraint": "Renamed X-Client-Token to X-Auth-Token and dropped legacy header",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "ext-04",
        "family": "extraction",
        "title": "Geospatial Coordinate Normalization",
        "prompt_text": (
            "Extract the latitude and longitude from: 'Station Bravo is positioned at 37 deg 48 min N, "
            "122 deg 24 min W at 42m elevation.' Express both as decimal degrees rounded to two decimals."
        ),
        "expected_constraint": "37.80, -122.40",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "ext-05",
        "family": "extraction",
        "title": "Contract Notice Window Extraction",
        "prompt_text": (
            "From the clause: 'Either party may terminate upon 45 calendar days written notice prior "
            "to renewal, except for material breach which requires 15 business days cure notice.' "
            "State both notice periods and their day types."
        ),
        "expected_constraint": "45 calendar days; 15 business days",
        "validation_status": VALIDATION_LABEL,
    },
    # Family 2: Reasoning (5 prompts)
    {
        "prompt_id": "rea-01",
        "family": "reasoning",
        "title": "Modular Arithmetic Remainder",
        "prompt_text": (
            "Compute the remainder when (17^3 + 29 * 14) is divided by 11. Show the intermediate "
            "modular reductions briefly and state the final integer remainder."
        ),
        "expected_constraint": "Final remainder in 0..10",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "rea-02",
        "family": "reasoning",
        "title": "Constraint Scheduling Ordering",
        "prompt_text": (
            "Four jobs (A, B, C, D) run sequentially without overlap. C must finish before A starts. "
            "B must run immediately after D. D cannot be first. List the single valid execution order."
        ),
        "expected_constraint": "C, D, B, A",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "rea-03",
        "family": "reasoning",
        "title": "Bayes Base-Rate Calibration",
        "prompt_text": (
            "A sensor defect occurs in 1% of units. A diagnostic test detects defective units 90% of "
            "the time and gives a false alarm on non-defective units 9% of the time. If a unit tests "
            "positive, what is the exact fraction of positive units that are truly defective?"
        ),
        "expected_constraint": "10/109 (~9.17%)",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "rea-04",
        "family": "reasoning",
        "title": "Graph Bipartite Cycle Check",
        "prompt_text": (
            "A simple undirected graph has vertices {1,2,3,4,5} and edges (1,2), (2,3), (3,4), (4,5), "
            "(5,1), and (1,3). Explain in two sentences whether the graph is bipartite and why."
        ),
        "expected_constraint": "Not bipartite due to odd cycle (e.g., 1-2-3)",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "rea-05",
        "family": "reasoning",
        "title": "Counterfactual Tank Flow Rate",
        "prompt_text": (
            "Pipe X fills a 120-liter tank in 6 hours; drain Y empties it in 10 hours. If both are "
            "open simultaneously starting from an empty tank, how many hours until the tank holds "
            "64 liters?"
        ),
        "expected_constraint": "8 hours (net 8 L/hr)",
        "validation_status": VALIDATION_LABEL,
    },
    # Family 3: Code (5 prompts)
    {
        "prompt_id": "cod-01",
        "family": "code",
        "title": "Python Run-Length Encoder",
        "prompt_text": (
            "Write a concise Python function `rle_encode(s: str) -> str` that compresses consecutive "
            "identical characters into `<char><count>` (e.g., `'aaabbc'` -> `'a3b2c1'`). Return only "
            "the function definition without markdown fences."
        ),
        "expected_constraint": "def rle_encode",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "cod-02",
        "family": "code",
        "title": "SQL Window Second Highest Value",
        "prompt_text": (
            "Write a single ANSI SQL query using `DENSE_RANK()` over table `readings(sensor_id, val)` "
            "to return `sensor_id` and `val` for the second-highest distinct `val` per `sensor_id`."
        ),
        "expected_constraint": "DENSE_RANK() OVER (PARTITION BY sensor_id ORDER BY val DESC)",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "cod-03",
        "family": "code",
        "title": "Monotonic Stack Next Greater Index",
        "prompt_text": (
            "Write a Python function `next_greater_indices(nums: list[int]) -> list[int]` that returns "
            "for each index the index of the next strictly greater element to its right, or -1 if none."
        ),
        "expected_constraint": "def next_greater_indices",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "cod-04",
        "family": "code",
        "title": "Floating-Point Kahan Summation",
        "prompt_text": (
            "Write a short Python function `kahan_sum(values)` implementing compensated Kahan summation "
            "for an iterable of floats. Keep it under 12 lines."
        ),
        "expected_constraint": "def kahan_sum",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "cod-05",
        "family": "code",
        "title": "POSIX Shell Safe Tempfile Trap",
        "prompt_text": (
            "Show a 4-line POSIX sh snippet that creates a temporary file with `mktemp` and registers "
            "an `EXIT` `trap` to remove it cleanly."
        ),
        "expected_constraint": "mktemp and trap",
        "validation_status": VALIDATION_LABEL,
    },
    # Family 4: Multilingual (5 prompts)
    {
        "prompt_id": "mul-01",
        "family": "multilingual",
        "title": "Spanish Technical Clause Translation",
        "prompt_text": (
            "Translate into formal Spanish: 'The calibration artifact must be verified before the "
            "automated worker resumes interrupted requests.' Output only the Spanish sentence."
        ),
        "expected_constraint": "Spanish sentence",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "mul-02",
        "family": "multilingual",
        "title": "German Compound Terminology Explanation",
        "prompt_text": (
            "Translate 'confidence interval under finite-sample exchangeability' into precise German "
            "statistical terminology and provide a one-sentence German definition."
        ),
        "expected_constraint": "Konfidenzintervall / Austauschbarkeit",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "mul-03",
        "family": "multilingual",
        "title": "French API Rate-Limit Notice",
        "prompt_text": (
            "Write a polite, one-sentence error message in French stating that the endpoint exceeded "
            "its per-minute request quota and should retry after 30 seconds."
        ),
        "expected_constraint": "30 secondes",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "mul-04",
        "family": "multilingual",
        "title": "Japanese Polite Status Summary",
        "prompt_text": (
            "Translate into polite Japanese (Desu/Masu form): 'All twenty validation prompts completed "
            "without network errors.' Output only the Japanese text."
        ),
        "expected_constraint": "Japanese script",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "mul-05",
        "family": "multilingual",
        "title": "Hindi Bilingual Glossary Pair",
        "prompt_text": (
            "Provide the Hindi translation in Devanagari script for the phrase 'Reference model "
            "verification passed within tested limits' followed by a literal English back-translation."
        ),
        "expected_constraint": "Devanagari + back-translation",
        "validation_status": VALIDATION_LABEL,
    },
    # Family 5: Formatting (4 prompts -> 24 total prompts)
    {
        "prompt_id": "fmt-01",
        "family": "formatting",
        "title": "Strict Single-Line JSON Object",
        "prompt_text": (
            "Output a single-line minified JSON object with keys `\"service\"`, `\"latency_ms\"`, and "
            "`\"healthy\"` set to `\"router-east\"`, `42`, and `true` respectively. No prose."
        ),
        "expected_constraint": '{"healthy":true,"latency_ms":42,"service":"router-east"}',
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "fmt-02",
        "family": "formatting",
        "title": "RFC4180 Three-Row CSV Table",
        "prompt_text": (
            "Output a 3-row CSV (1 header + 2 data rows) with columns `metric,value,unit` for "
            "`cvm_stat,0.184,dimensionless` and `duration,12.5,seconds`. No markdown fences."
        ),
        "expected_constraint": "metric,value,unit",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "fmt-03",
        "family": "formatting",
        "title": "Key-Value INI Section Block",
        "prompt_text": (
            "Format the settings `host=127.0.0.1`, `port=8000`, `workers=1` under an INI section header "
            "`[server]` with no extra commentary."
        ),
        "expected_constraint": "[server]",
        "validation_status": VALIDATION_LABEL,
    },
    {
        "prompt_id": "fmt-04",
        "family": "formatting",
        "title": "Fixed-Bullet Audit Checklist",
        "prompt_text": (
            "Write exactly three bullet points, each starting with `- [x] ` and containing five words, "
            "summarizing pre-flight checks for a database migration."
        ),
        "expected_constraint": "Three '- [x] ' lines",
        "validation_status": VALIDATION_LABEL,
    },
]


def parse_probe_import_payload(fmt: str, raw_content: str) -> List[ProbePromptItem]:
    """Parse and validate JSON or JSONL probe import content with strict bounds and duplicate checks."""
    if not raw_content or not raw_content.strip():
        raise ValueError("Probe import content cannot be empty.")
    if len(raw_content) > 500_000:
        raise ValueError("Probe import payload exceeds maximum size of 500,000 characters.")

    fmt_clean = (fmt or "json").strip().lower()
    raw_items: List[Dict[str, Any]] = []

    if fmt_clean == "json":
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc
        if isinstance(parsed, dict) and "prompts" in parsed:
            parsed = parsed["prompts"]
        if not isinstance(parsed, list):
            raise ValueError("JSON probe import must be a list of prompt objects or {'prompts': [...]}.")
        raw_items = parsed
    elif fmt_clean == "jsonl":
        lines = [ln.strip() for ln in raw_content.splitlines() if ln.strip()]
        for idx, line in enumerate(lines, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {idx} must be a JSON object.")
            raw_items.append(obj)
    else:
        raise ValueError("Unsupported probe import format. Use 'json' or 'jsonl'.")

    if not raw_items:
        raise ValueError("Probe import must contain at least 1 prompt.")
    if len(raw_items) > 200:
        raise ValueError(f"Probe import exceeds maximum of 200 prompts (got {len(raw_items)}).")

    validated_items: List[ProbePromptItem] = []
    seen_ids = set()
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry #{idx} must be an object.")
        if "validation_status" not in item or not item["validation_status"]:
            item = dict(item)
            item["validation_status"] = VALIDATION_LABEL
        prompt_obj = ProbePromptItem(**item)
        if prompt_obj.prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt_id '{prompt_obj.prompt_id}' detected in import.")
        seen_ids.add(prompt_obj.prompt_id)
        validated_items.append(prompt_obj)

    return validated_items
