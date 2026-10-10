import sys
from pathlib import Path
ROOT = Path("/Users/benrosen/Documents/prediction_models/nfl-player-projections"); sys.path.insert(0, str(ROOT / "scripts"))
import audit_serving_leakage as A
work = Path(sys.argv[1]); tree = work / "tree"
A.build_scratch_tree(tree); A.copy_database(tree / "data" / "nfl_data.db")
for w in (1, 2, 3, 4):
    A.run_capture(tree, work / f"w{w}.pkl", 2026, w, tree); print("captured week", w, flush=True)
