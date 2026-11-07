"""Compatibility shim; use modelling.sensitivity.alternative_definitions."""
from modelling.sensitivity.alternative_definitions import (
    Path,
    active_input_paths,
    argparse,
    canonical_input_hashes,
    evaluate,
    evidence_dir,
    json,
    load_master,
    main,
    make_episodes,
    np,
    pd,
    save_json,
    sha256,
    targetable_ids,
)

if __name__ == "__main__":
    main()
