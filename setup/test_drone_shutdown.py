"""Exercise the launcher's actual shutdown function with a small child process."""

from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class ShutdownTest(unittest.TestCase):
    def test_launcher_does_not_return_before_brain_shutdown_saves_map(self):
        source = (Path(__file__).resolve().parents[1] / "simulation/run_drone.sh").read_text()
        finish = source[source.index("finish() {"):source.index("trap finish EXIT")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "brain.py"
            child.write_text('''import signal, time
from pathlib import Path
def finish(*_):
    time.sleep(0.05)
    Path("map-saved").write_text("complete")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, finish)
Path("ready").touch()
while True:
    time.sleep(0.05)
''')
            shell = "\n".join([
                "set -eu",
                finish,
                "seq() { printf '1\\n'; }",
                "sleep() { command sleep 0.1; }",
                "setsid " + shlex.quote(sys.executable) + " " + shlex.quote(str(child)) + " &",
                "BRAIN_PID=$!",
                "trap 'kill -KILL -- -\"${BRAIN_PID}\" 2>/dev/null || true' EXIT",
                "while [ ! -f ready ]; do command sleep 0.01; done",
                "finish",
                "test -f map-saved",
            ])
            result = subprocess.run(["bash", "-c", shell], cwd=root, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "map-saved").read_text(), "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
