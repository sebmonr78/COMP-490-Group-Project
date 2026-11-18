"""Check dependency discovery without changing the real process environment."""

import json
from pathlib import Path
import tempfile
import unittest

import server_environment


class ServerEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = self.root / ".server/environment.json"
        self.settings.parent.mkdir()
        self.conda = self.root / "deps/bin/conda"
        self.conda.parent.mkdir(parents=True)
        self.conda.write_text("#!/bin/sh\n")
        self.conda.chmod(0o755)

    def write(self, **values):
        self.settings.write_text(json.dumps(values))

    def test_missing_settings_leave_manual_setup_unchanged(self):
        environment = {"PATH": "/usr/bin"}
        server_environment.configure(self.root, environment)
        self.assertEqual(environment, {"PATH": "/usr/bin"})

    def test_dependency_locations_are_loaded_without_shell_activation(self):
        self.write(conda_executable=str(self.conda), dataset_root="/data/hssd")
        environment = {"PATH": "/usr/bin"}
        server_environment.configure(self.root, environment)
        self.assertEqual(environment["CONDA_EXE"], str(self.conda))
        self.assertEqual(environment["HABITAT_DATA_PATH"], "/data/hssd")
        self.assertEqual(environment["PATH"].split(":"), [str(self.conda.parent), "/usr/bin"])

    def test_explicit_user_settings_take_precedence(self):
        self.write(conda_executable=str(self.conda), dataset_root="/new", wsl_rendering=True)
        environment = {"CONDA_EXE": "/user/bin/conda", "HABITAT_DATA_PATH": "/user/data",
                       "GALLIUM_DRIVER": "user-renderer", "PATH": "/usr/bin"}
        server_environment.configure(self.root, environment)
        self.assertEqual(environment["CONDA_EXE"], "/user/bin/conda")
        self.assertEqual(environment["HABITAT_DATA_PATH"], "/user/data")
        self.assertEqual(environment["GALLIUM_DRIVER"], "user-renderer")

    def test_repeated_loading_does_not_duplicate_path(self):
        self.write(conda_executable=str(self.conda))
        environment = {"PATH": "/usr/bin"}
        server_environment.configure(self.root, environment)
        before = dict(environment)
        server_environment.configure(self.root, environment)
        self.assertEqual(environment, before)

    def test_invalid_settings_warn_and_leave_environment_unchanged(self):
        for contents in ("{", "[]", '{"dataset_root": ["invalid"]}'):
            self.settings.write_text(contents)
            environment = {"PATH": "/usr/bin"}
            with self.assertWarns(UserWarning):
                server_environment.configure(self.root, environment)
            self.assertEqual(environment, {"PATH": "/usr/bin"})

    def test_deleted_conda_is_not_added_to_path(self):
        self.write(conda_executable=str(self.conda))
        self.conda.unlink()
        environment = {"PATH": "/usr/bin"}
        server_environment.configure(self.root, environment)
        self.assertEqual(environment, {"PATH": "/usr/bin"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
