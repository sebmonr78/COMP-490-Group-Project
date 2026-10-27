"""Drive the drone around the house and watch it map it.

    bash run_drone.sh            the brain and this window together
    python demo/run_drone.py     this window only (--no-brain, or a brain
                                 already started)

The window's code is in delta/window/ (see its __init__ for what is where);
the brain's in delta/brain/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delta.window.app import main  # noqa: E402

if __name__ == "__main__":
    main()
