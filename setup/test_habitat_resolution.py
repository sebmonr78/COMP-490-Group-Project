"""Exercise Habitat's dependency constraints using pip and local test wheels.

These regression tests use metadata-only stand-ins in temporary virtual
environments. They never download packages, access CUDA or change the server.
"""

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv
import zipfile

import server_dependencies as setup


def wheel(directory, name, version, requirements=()):
    normalized = name.lower().replace("-", "_")
    info = normalized + "-" + version + ".dist-info"
    contents = {
        info + "/METADATA": "Metadata-Version: 2.1\nName: %s\nVersion: %s\n%s\n" % (
            name, version, "".join("Requires-Dist: " + item + "\n" for item in requirements)),
        info + "/WHEEL": "Wheel-Version: 1.0\nGenerator: dependency-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        normalized + "/__init__.py": "__version__ = " + repr(version) + "\n",
    }
    records = io.StringIO()
    writer = csv.writer(records, lineterminator="\n")
    for path in contents:
        writer.writerow([path, "", ""])
    writer.writerow([info + "/RECORD", "", ""])
    contents[info + "/RECORD"] = records.getvalue()
    destination = directory / (normalized + "-" + version + "-py3-none-any.whl")
    with zipfile.ZipFile(destination, "w") as archive:
        for path, data in contents.items():
            archive.writestr(path, data)
    return destination


class HabitatResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.wheels = cls.root / "wheels"
        cls.wheels.mkdir()
        cls.environment = cls.root / "venv"
        venv.EnvBuilder(with_pip=True).create(cls.environment)
        cls.python = cls.environment / "bin/python"
        cls.env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1",
                       PIP_FIND_LINKS=str(cls.wheels), PIP_DISABLE_PIP_VERSION_CHECK="1",
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        cls.env.pop("PYTHONPATH", None)
        cls.env.pop("PYTHONHOME", None)
        for key in ("PIP_CONSTRAINT", "PIP_REQUIREMENT", "PIP_FORCE_REINSTALL",
                    "PIP_IGNORE_INSTALLED", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
            cls.env.pop(key, None)
        for name, version, dependencies in [
            ("Pillow", "10.4.0", ()), ("Pillow", "11.3.0", ()),
            ("numpy", "1.26.4", ()), ("huggingface-hub", "0.36.2", ()),
            ("torch", "2.8.0", ()),
            ("pygame", "2.6.1", ()), ("psutil", "7.2.2", ()),
            ("opencv-python", "4.11.0.86", ("numpy>=1.23.5",)),
            ("opencv-python", "4.13.0.92", ("numpy>=2",)),
            ("habitat-sim", "0.3.3", ("Pillow==10.4.0", "numpy<2")),
            ("habitat-lab", "0.3.3", ("Pillow>=9", "numpy", "torch>=2")),
            ("habitat-baselines", "0.3.3", ("habitat-lab==0.3.3",)),
            ("habitat-hitl", "0.3.3", ("habitat-lab==0.3.3",)),
            ("incompatible-test-package", "1.0", ("Pillow>=11.3",)),
        ]:
            wheel(cls.wheels, name, version, dependencies)
        cls.constraints = cls.root / "habitat-constraints.txt"
        cls.constraints.write_text("\n".join(setup.HABITAT_REQUIREMENTS) + "\n")
        cls.pip("install", "--no-deps", "habitat-sim==0.3.3", "Pillow==11.3.0", "numpy==1.26.4", "torch==2.8.0")
        broken = cls.pip("check", expect_success=False)
        if broken.returncode == 0 or "10.4.0" not in broken.stdout:
            raise AssertionError("The test did not reproduce the reported Pillow conflict: " + broken.stdout)

    @classmethod
    def pip(cls, *args, expect_success=True):
        result = subprocess.run([str(cls.python), "-m", "pip"] + list(args), env=cls.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=60)
        if expect_success and result.returncode:
            raise AssertionError(result.stdout)
        return result

    @classmethod
    def versions(cls):
        result = subprocess.run([str(cls.python), "-c",
                                 "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('Pillow','torch')}))"],
                                env=cls.env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_1_repair_downgrades_pillow_without_reinstalling_compatible_torch(self):
        torch = next((self.environment / "lib").glob("python*/site-packages/torch/__init__.py"))
        before = torch.read_bytes(), torch.stat().st_mtime_ns
        command = ["install", "--constraint", str(self.constraints), "--upgrade-strategy", "only-if-needed"]
        command += list(setup.HABITAT_REQUIREMENTS)
        command += list(setup.HABITAT_WINDOW_REQUIREMENTS)
        command += ["habitat-lab==0.3.3", "habitat-baselines==0.3.3", "habitat-hitl==0.3.3"]
        self.pip(*command)
        self.assertEqual(self.versions(), {"Pillow": "10.4.0", "torch": "2.8.0"})
        self.pip("check")
        self.assertEqual((torch.read_bytes(), torch.stat().st_mtime_ns), before)
        self.pip(*command)
        self.assertEqual((torch.read_bytes(), torch.stat().st_mtime_ns), before)

    def test_3_window_install_preserves_existing_numpy_and_selects_compatible_opencv(self):
        constraints = self.root / "window-constraints.txt"
        constraints.write_text("\n".join(setup.HABITAT_REQUIREMENTS) + "\nnumpy==1.26.4\n")
        numpy_file = next((self.environment / "lib").glob("python*/site-packages/numpy/__init__.py"))
        before = numpy_file.read_bytes(), numpy_file.stat().st_mtime_ns
        self.pip("install", "--constraint", str(constraints), *setup.HABITAT_WINDOW_REQUIREMENTS)
        self.pip("check")
        result = subprocess.run([str(self.python), "-c",
                                 "from importlib.metadata import version; print(version('opencv-python')); import pygame, psutil"],
                                env=self.env, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "4.11.0.86")
        self.assertEqual((numpy_file.read_bytes(), numpy_file.stat().st_mtime_ns), before)

    def test_2_conflicting_dependency_is_rejected_without_changing_pillow(self):
        self.pip("install", "--constraint", str(self.constraints), "Pillow==10.4.0")
        before = self.versions()
        result = self.pip("install", "--constraint", str(self.constraints),
                          "incompatible-test-package==1.0", expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ResolutionImpossible", result.stdout)
        self.assertEqual(self.versions(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
