"""Reproduce the GTSAM/NumPy metadata conflict using local test wheels only."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import venv

import server_dependencies as setup
from test_habitat_resolution import wheel


class RtabmapResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.wheels = cls.root / "wheels"
        cls.wheels.mkdir()
        cls.prefix = cls.root / "rtabmap"
        venv.EnvBuilder(with_pip=True).create(cls.prefix)
        cls.python = cls.prefix / "bin/python"
        records = cls.prefix / "conda-meta"
        records.mkdir()
        (records / "ros-humble-rtabmap-0.22.1.json").write_text(
            json.dumps({"name": "ros-humble-rtabmap", "version": "0.22.1"}))
        cls.env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1",
                       PIP_FIND_LINKS=str(cls.wheels), PIP_DISABLE_PIP_VERSION_CHECK="1",
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        for key in ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "PIP_REQUIREMENT",
                    "PIP_FORCE_REINSTALL", "PIP_IGNORE_INSTALLED", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
            cls.env.pop(key, None)
        wheel(cls.wheels, "numpy", "2.5.4")
        wheel(cls.wheels, "numpy", "1.26.4")
        wheel(cls.wheels, "gtsam", "4.2.2", ("numpy>=1.11,<2.0.0",))

    def pip(self, *args):
        return subprocess.run([str(self.python), "-m", "pip"] + list(args), env=self.env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=60)

    def native_installer(self):
        installer = setup.Installer(setup.ROOT, setup.arguments(["--check"]))
        installer.conda = "/test/conda"
        installer.say = mock.Mock()
        installer.python_probe = mock.Mock(side_effect=AssertionError("Unexpected target Python validation"))
        records = [{"name": "ros-humble-rtabmap", "version": "0.22.1"}]
        records += [{"name": name, "version": "1"} for name in ("cxx-compiler", "cmake", "make", "pkg-config")]
        records += [{"name": "python", "version": "3.14.0"}, {"name": "numpy", "version": "2.5.4"},
                    {"name": "gtsam", "version": "4.2.2"}]

        def command(argv, **kwargs):
            argv = [str(part) for part in argv]
            if argv == ["/test/conda", "list", "--prefix", str(self.prefix), "--json"]:
                return 0, json.dumps(records)
            if argv[:5] == ["/test/conda", "run", "--no-capture-output", "--prefix", str(self.prefix)]:
                self.assertIn(argv[5], ("cmake", "make", "pkg-config"))
                self.assertEqual(argv[6:], ["--version"])
                return 0, argv[5] + " version test"
            raise AssertionError("Unexpected dependency command: " + repr(argv))

        installer.run = mock.Mock(side_effect=command)
        return installer

    def test_unused_python_binding_conflict_does_not_block_native_rtabmap(self):
        installed = self.pip("install", "--no-deps", "gtsam==4.2.2", "numpy==2.5.4")
        self.assertEqual(installed.returncode, 0, installed.stdout)
        result = self.pip("check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("2.5.4", result.stdout)
        pin = self.prefix / "conda-meta/pinned"
        pin.write_text("python=3.14\n")
        installer = self.native_installer()
        ok, detail = installer.rtabmap_probe(self.prefix)
        self.assertTrue(ok, detail)
        self.assertEqual(pin.read_text(), "python=3.14\n")
        installer.python_probe.assert_not_called()

    def test_python_314_pinned_native_environment_is_reused_without_solving(self):
        installer = self.native_installer()
        installer.conda_prefix = mock.Mock(return_value=self.prefix)
        installer.setup_conda_env("rtabmap", setup.RTABMAP_REQUIREMENTS, None, check=installer.rtabmap_probe)
        self.assertEqual(installer.failures, [])
        commands = [[str(part) for part in call.args[0]] for call in installer.run.call_args_list]
        self.assertFalse(any("install" in argv or "create" in argv or "remove" in argv for argv in commands))
        self.assertFalse(any("pip" in argv or "--no-pin" in argv for argv in commands))
        installer.python_probe.assert_not_called()

    def test_missing_native_package_still_fails_validation(self):
        installer = self.native_installer()
        installer.run = mock.Mock(return_value=(0, "[]"))
        ok, detail = installer.rtabmap_probe(self.prefix)
        self.assertFalse(ok)
        self.assertIn("native library 0.22.1", detail)

    def test_broken_native_tool_still_fails_validation(self):
        installer = self.native_installer()
        original = installer.run.side_effect

        def command(argv, **kwargs):
            if "run" in argv:
                return 127, "native shared library could not be loaded"
            return original(argv, **kwargs)

        installer.run.side_effect = command
        ok, detail = installer.rtabmap_probe(self.prefix)
        self.assertFalse(ok)
        self.assertIn("native tool", detail)

    def test_failed_native_bridge_is_not_hidden_by_package_validation(self):
        installer = self.native_installer()
        installer.run = mock.Mock(return_value=(127, "libgtsam.so could not be loaded"))
        installer.setup_bridge(self.prefix)
        self.assertIn("RTAB-Map bridge", installer.failures)

    def test_working_native_bridge_is_reused(self):
        installer = self.native_installer()
        installer.run = mock.Mock(return_value=(0, "rtabmap_bridge, RTAB-Map 0.22.1"))
        installer.setup_bridge(self.prefix)
        self.assertEqual(installer.failures, [])
        self.assertEqual(installer.run.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
