"""
Print candidate_selection's per-slot candidates for the probe harness's demo
stanza (llm_probe's MASK_WORDS, SOURCE_AUTHOR -> TARGET_AUTHOR).

Usage:
    uv run python -m llm_infill.show_slots
"""

import json
import logging

from common.masking.candidate_selection import build_slots, log_slot
from llm_infill.llm_probe import MASK_WORDS, SOURCE_AUTHOR, TARGET_AUTHOR, load_stanza


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    stanza = load_stanza()
    slots = build_slots(stanza, MASK_WORDS, SOURCE_AUTHOR, TARGET_AUTHOR)
    print(json.dumps(slots, ensure_ascii=False, indent=2))
    for slot in slots:
        print(log_slot(slot))
