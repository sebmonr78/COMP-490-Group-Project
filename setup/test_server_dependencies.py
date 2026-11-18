"""Tests for dependency repair, rebuild consent and application-data isolation.

    python3 -m unittest discover -s setup -p 'test_*.py'

No test installs packages, invokes sudo or contacts a package server.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import server_dependencies as setup


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        script = self.root / "simulation/setup/setup_brain.sh"
        script.parent.mkdir(parents=True)
        shutil.copy2(setup.ROOT / "simulation/setup/setup_brain.sh", script)
        self.installer = setup.Installer(self.root, setup.arguments([]), ask=lambda _: "repair")
        self.installer.say = mock.Mock()
        self.path = self.root / "app/venv"
        self.path.mkdir(parents=True)

    def test_healthy_existing_environment_is_skipped_without_prompt(self):
        self.installer.ask = mock.Mock(side_effect=AssertionError("Unexpected prompt"))
        repair, rebuild = mock.Mock(), mock.Mock()
        self.installer.ensure("website", self.path, lambda: (True, "ready"), repair, rebuild)
        repair.assert_not_called()
        rebuild.assert_not_called()

    def test_repair_is_rechecked_and_rebuild_is_not_called(self):
        probe = mock.Mock(side_effect=[(False, "wrong version"), (True, "ready")])
        repair, rebuild = mock.Mock(), mock.Mock()
        self.installer.ensure("website", self.path, probe, repair, rebuild)
        repair.assert_called_once_with()
        rebuild.assert_not_called()
        self.assertEqual(probe.call_count, 2)

    def test_explicit_rebuild_does_not_use_repair(self):
        self.installer.ask = lambda _: "rebuild"
        probe = mock.Mock(side_effect=[(False, "old Python"), (True, "ready")])
        repair, rebuild = mock.Mock(), mock.Mock()
        self.installer.ensure("website", self.path, probe, repair, rebuild, can_repair=False)
        rebuild.assert_called_once_with()
        repair.assert_not_called()

    def test_cancel_performs_no_mutation(self):
        self.installer.ask = lambda _: "cancel"
        repair, rebuild = mock.Mock(), mock.Mock()
        with self.assertRaises(setup.SetupError):
            self.installer.ensure("website", self.path, lambda: (False, "old"), repair, rebuild)
        repair.assert_not_called()
        rebuild.assert_not_called()

    def test_check_mode_never_prompts_repairs_or_creates_state(self):
        self.installer.args.check = True
        self.installer.ask = mock.Mock(side_effect=AssertionError("Unexpected prompt"))
        repair, rebuild = mock.Mock(), mock.Mock()
        self.installer.ensure("website", self.path, lambda: (False, "missing"), repair, rebuild)
        self.assertEqual(self.installer.failures, ["website"])
        repair.assert_not_called()
        rebuild.assert_not_called()
        self.assertFalse((self.root / ".server").exists())

    def test_yes_repairs_but_does_not_consent_to_incompatible_interpreter_rebuild(self):
        self.installer.args.yes = True
        self.assertEqual(self.installer.choice("website", self.path), "repair")
        with self.assertRaisesRegex(setup.SetupError, "incompatible interpreter"):
            self.installer.choice("website", self.path, can_repair=False)

    def test_rebuild_flag_still_requires_explicit_rebuild_answer(self):
        self.installer.args.rebuild = True
        self.installer.ask = lambda _: ""
        with self.assertRaises(setup.SetupError):
            self.installer.choice("website", self.path)

    def test_failed_post_install_probe_is_not_reported_as_ready(self):
        repair = mock.Mock()
        with self.assertRaisesRegex(setup.SetupError, "still failed validation"):
            self.installer.ensure("website", self.path, lambda: (False, "broken import"), repair, mock.Mock())
        self.assertFalse(any("validated" in str(call) for call in self.installer.say.call_args_list))

    def test_rebuild_removes_only_exact_project_virtual_environment(self):
        (self.path / "pyvenv.cfg").write_text("home = /dependency/python\n")
        old = self.path / "old-package"
        old.write_text("old")
        protected = [self.root / name for name in ["database/delta.db", "database/saved_runs/map.png",
                                                  "simulation/outputs/runs/web-1/memory/grid.npz", "app/main.py"]]
        for file in protected:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"keep exactly")
        self.installer.ask = lambda _: "rebuild"
        self.installer.python_probe = mock.Mock(side_effect=[(False, "old"), (True, "ready")])
        self.installer.run = mock.Mock(return_value=(0, ""))
        self.installer.setup_venv("app/venv", "app/requirements.txt", "fastapi", "/python")
        self.assertFalse(old.exists())
        for file in protected:
            self.assertEqual(file.read_bytes(), b"keep exactly")

    def test_unrecognized_folder_is_never_deleted_by_venv_rebuild(self):
        (self.path / "user-file").write_text("preserve")
        self.installer.ask = lambda _: "rebuild"
        self.installer.python_probe = mock.Mock(return_value=(False, "missing"))
        with self.assertRaisesRegex(setup.SetupError, "not a recognized"):
            self.installer.setup_venv("app/venv", "app/requirements.txt", "fastapi", "/python")
        self.assertEqual((self.path / "user-file").read_text(), "preserve")

    def test_symlink_virtual_environment_is_never_deleted(self):
        self.path.rmdir()
        external = self.root / "external"
        external.mkdir()
        (external / "pyvenv.cfg").write_text("home = /other")
        self.path.symlink_to(external, target_is_directory=True)
        self.installer.ask = lambda _: "rebuild"
        self.installer.python_probe = mock.Mock(return_value=(False, "broken"))
        with self.assertRaisesRegex(setup.SetupError, "Refusing to delete"):
            self.installer.setup_venv("app/venv", "app/requirements.txt", "fastapi", "/python")
        self.assertTrue((external / "pyvenv.cfg").exists())

    def test_conda_removal_refuses_base_and_unknown_environment(self):
        self.installer.run = mock.Mock()
        with self.assertRaises(setup.SetupError):
            self.installer.remove_conda_env("base", self.root)
        with self.assertRaises(setup.SetupError):
            self.installer.remove_conda_env("brain", self.root / "brain")
        self.installer.run.assert_not_called()

    def test_active_conda_environment_cannot_be_rebuilt(self):
        prefix = self.root / "brain"
        (prefix / "conda-meta").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"CONDA_PREFIX": str(prefix)}):
            with self.assertRaisesRegex(setup.SetupError, "Deactivate"):
                self.installer.remove_conda_env("brain", prefix)

    def test_active_virtual_environment_cannot_be_rebuilt(self):
        (self.path / "pyvenv.cfg").write_text("home = /dependency/python\n")
        self.installer.ask = lambda _: "rebuild"
        self.installer.python_probe = mock.Mock(return_value=(False, "old"))
        with mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(self.path)}):
            with self.assertRaisesRegex(setup.SetupError, "Deactivate"):
                self.installer.setup_venv("app/venv", "app/requirements.txt", "fastapi", "/python")
        self.assertTrue((self.path / "pyvenv.cfg").is_file())

    def test_unverified_bootstrap_download_is_never_executed(self):
        self.installer.find_conda = mock.Mock(return_value=None)
        self.installer.run = mock.Mock()
        with mock.patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong installer")):
            with self.assertRaisesRegex(setup.SetupError, "checksum"):
                self.installer.setup_conda()
        self.installer.run.assert_not_called()
        self.assertFalse((self.root / ".server/conda").exists())

    def test_gpu_check_rejects_old_driver_but_accepts_compatible_driver(self):
        self.installer.run = mock.Mock(return_value=(0, "NVIDIA test GPU, 550.90.07"))
        with self.assertRaisesRegex(setup.SetupError, "570"):
            self.installer.check_gpu()
        self.installer.run.return_value = (0, "NVIDIA test GPU, 580.65.06")
        self.installer.check_gpu()

    def test_existing_manually_installed_conda_is_reused(self):
        candidate = self.root / "conda/bin/conda"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("executable")
        self.installer.run = mock.Mock(return_value=(0, "conda 26.9.4"))
        with mock.patch.dict(os.environ, {"CONDA_EXE": str(candidate)}):
            self.assertEqual(self.installer.find_conda(), str(candidate))

    def test_runtime_settings_contain_only_dependency_locations(self):
        config = self.root / "simulation/config/project.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"dataset": {"default_root": "~/existing-hssd", "configuration_file": "scene.json"}}))
        self.installer.state.mkdir()
        self.installer.conda = "/dependency/conda"
        self.installer.write_environment()
        data = json.loads((self.installer.state / "environment.json").read_text())
        self.assertEqual(data["conda_executable"], "/dependency/conda")
        self.assertFalse((self.root / "database").exists())
        self.assertFalse((self.root / "simulation/outputs").exists())

    def test_pins_match_the_existing_brain_setup(self):
        pins = setup.read_pins(self.root)
        self.assertEqual(pins["TORCH"], "2.10.0")
        self.assertEqual(pins["XFORMERS"], "0.0.35")
        self.assertEqual(len(pins["DA3_COMMIT"]), 40)
        self.assertEqual(pins["ULTRALYTICS"], "8.4.177")
        self.assertEqual(pins["PIPER"], "1.8.0")

    def habitat_install_command(self):
        self.installer.run = mock.Mock(return_value=(0, ""))
        prefix = self.root / "envs/habitat"
        self.installer.habitat_packages(prefix)
        return [str(part) for part in self.installer.run.call_args.args[0]]

    def test_habitat_pins_pillow_and_includes_simulator_in_resolution(self):
        command = self.habitat_install_command()
        constraints = Path(command[command.index("--constraint") + 1]).read_text().splitlines()
        self.assertIn("Pillow==10.4.0", constraints)
        self.assertIn("Pillow==10.4.0", command)
        self.assertIn("habitat-sim==0.3.3", command)
        self.assertEqual(command[:4], [str(self.root / "envs/habitat/bin/python"), "-m", "pip", "install"])

    def test_habitat_preserves_compatible_installed_dependencies(self):
        command = self.habitat_install_command()
        self.assertNotIn("--force-reinstall", command)
        self.assertNotIn("--ignore-installed", command)
        self.assertEqual(command[command.index("--upgrade-strategy") + 1], "only-if-needed")

    def test_habitat_requests_supported_editable_compatibility_mode(self):
        command = self.habitat_install_command()
        self.assertEqual(command[command.index("--config-settings") + 1], "editable_mode=compat")
        self.assertEqual(command.count("-e"), 3)
        self.assertNotIn("--use-pep517", command)

    def test_missing_window_packages_do_not_replace_working_habitat_lab_or_numpy(self):
        prefix = self.root / "envs/habitat"
        self.installer.conda_python_probe = mock.Mock(return_value=(True, "ready"))
        self.installer.run = mock.Mock(return_value=(0, "1.26.4"))
        self.installer.habitat_packages(prefix)
        commands = [[str(part) for part in call.args[0]] for call in self.installer.run.call_args_list]
        install = commands[-1]
        self.assertIn("pygame", install)
        self.assertIn("psutil", install)
        self.assertIn("opencv-python", install)
        self.assertIn("numpy==1.26.4", install)
        self.assertIn("Pillow==10.4.0", install)
        self.assertNotIn("-e", install)
        self.assertFalse(any(command[0] == "git" for command in commands))
        self.assertFalse((self.installer.state / "sources").exists())

    def test_pip_only_repair_skips_conda_solve_for_a_working_core(self):
        prefix = self.root / "envs/habitat"
        (prefix / "conda-meta").mkdir(parents=True)
        self.installer.conda = "/dependency/conda"
        self.installer.conda_prefix = mock.Mock(return_value=prefix)
        self.installer.python_probe = mock.Mock(side_effect=[(False, "No module named pygame"), (True, "ready")])
        self.installer.run = mock.Mock(side_effect=AssertionError("An existing native environment must not be solved again for pygame"))
        packages = mock.Mock()
        self.installer.setup_conda_env("habitat", ("python=3.9", "habitat-sim=0.3.3"), setup.HABITAT_PROBE,
                                       after=packages, core_check=lambda _: (True, "core ready"))
        packages.assert_called_once_with(prefix)
        self.installer.run.assert_not_called()

    def test_broken_core_still_runs_conda_repair_before_pip_packages(self):
        prefix = self.root / "envs/brain"
        (prefix / "conda-meta").mkdir(parents=True)
        self.installer.conda = "/dependency/conda"
        self.installer.conda_prefix = mock.Mock(return_value=prefix)
        self.installer.python_probe = mock.Mock(side_effect=[(False, "wrong Python"), (True, "ready")])
        self.installer.run = mock.Mock(return_value=(0, ""))
        packages = mock.Mock()
        self.installer.setup_conda_env("brain", ("python=3.10", "pip"), setup.BRAIN_PROBE,
                                       after=packages, core_check=lambda _: (False, "wrong Python"))
        self.assertEqual(self.installer.run.call_args.args[0][1], "install")
        self.assertIn("python=3.10", self.installer.run.call_args.args[0])
        packages.assert_called_once_with(prefix)

    def test_configured_conda_root_is_preferred(self):
        candidate = self.root / "custom-conda/bin/conda"
        candidate.parent.mkdir(parents=True)
        candidate.touch()
        self.installer.run = mock.Mock(return_value=(0, "conda 26.9.4"))
        with mock.patch.dict(os.environ, {"CONDA_ROOT": str(candidate.parent.parent), "CONDA_EXE": "/different/conda"}):
            self.assertEqual(self.installer.find_conda(), str(candidate))

    def test_dataset_root_is_used_without_creating_or_downloading_data(self):
        config = self.root / "simulation/config/project.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"dataset": {"default_root": "~/default", "configuration_file": "scene.json"}}))
        self.installer.state.mkdir()
        dataset = self.root / "existing-data"
        with mock.patch.dict(os.environ, {"HABITAT_DATA_ROOT": str(dataset)}):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HABITAT_DATA_PATH", None)
                self.installer.write_environment()
        saved = json.loads((self.installer.state / "environment.json").read_text())
        self.assertEqual(saved["dataset_root"], str(dataset / "versioned_data/hssd-hab"))
        self.assertFalse(dataset.exists())

    def test_brain_selects_exact_cuda_build_without_blanket_reinstallation(self):
        self.installer.run = mock.Mock(return_value=(0, ""))
        self.installer.brain_packages(self.root / "envs/brain")
        commands = [[str(part) for part in call.args[0]] for call in self.installer.run.call_args_list]
        self.assertIn("torch==2.10.0+cu128", commands[0])
        self.assertIn("torchvision==0.25.0+cu128", commands[0])
        self.assertFalse(any("--force-reinstall" in command for command in commands))
        constraints = (self.installer.state / "brain-constraints.txt").read_text()
        self.assertIn("torch==2.10.0+cu128", constraints)
        self.assertNotIn("Pillow", constraints)
        self.assertIn("ultralytics==8.4.177", commands[1])
        self.assertIn("piper-tts==1.8.0", commands[1])
        self.assertIn("opencv-python", commands[1])

    def test_rtabmap_creation_uses_original_native_dependency_recipe(self):
        prefix = self.root / "envs/rtabmap"
        self.installer.conda = "/dependency/conda"
        self.installer.conda_prefix = mock.Mock(return_value=prefix)
        native = mock.Mock(side_effect=[(False, "missing"), (True, "ready")])
        self.installer.python_probe = mock.Mock(side_effect=AssertionError("Python is not used for native RTAB-Map checks"))
        self.installer.run = mock.Mock(return_value=(0, ""))
        self.installer.setup_conda_env("rtabmap", setup.RTABMAP_REQUIREMENTS, None, check=native)
        command = [str(part) for part in self.installer.run.call_args.args[0]]
        self.assertEqual(command[:4], ["/dependency/conda", "create", "--prefix", str(prefix)])
        self.assertNotIn("numpy>=1.11,<2", command)
        self.assertNotIn("pip", command)
        self.assertFalse(any(part.startswith("python=") for part in command))
        self.assertIn("ros-humble-rtabmap=0.22.1", command)
        self.assertNotIn("--force-reinstall", command)
        self.assertNotIn("--no-pin", command)
        self.installer.python_probe.assert_not_called()

    def test_rtabmap_repair_does_not_add_numpy_or_override_python_pins(self):
        prefix = self.root / "envs/rtabmap"
        (prefix / "conda-meta").mkdir(parents=True)
        self.installer.conda = "/dependency/conda"
        self.installer.conda_prefix = mock.Mock(return_value=prefix)
        pin = prefix / "conda-meta/pinned"
        pin.write_text("python=3.14\n")
        native = mock.Mock(side_effect=[(False, "native build tool missing"), (True, "ready")])
        self.installer.python_probe = mock.Mock(side_effect=AssertionError("Unexpected Python check"))
        self.installer.run = mock.Mock(return_value=(0, ""))
        self.installer.setup_conda_env("rtabmap", setup.RTABMAP_REQUIREMENTS, None, check=native)
        command = [str(part) for part in self.installer.run.call_args.args[0]]
        self.assertEqual(command[:4], ["/dependency/conda", "install", "--prefix", str(prefix)])
        self.assertFalse(any(part.startswith(("numpy", "python", "pip")) for part in command))
        self.assertNotIn("--no-pin", command)
        self.assertFalse(any("remove" in call.args[0] for call in self.installer.run.call_args_list))
        self.assertEqual(native.call_count, 2)
        self.assertEqual(pin.read_text(), "python=3.14\n")
        self.installer.python_probe.assert_not_called()

    def test_venv_repair_does_not_force_reinstall_compatible_packages(self):
        python = self.path / "bin/python"
        python.parent.mkdir()
        python.touch()
        self.installer.python_probe = mock.Mock(side_effect=[(False, "missing dependency"), (True, "ready")])
        self.installer.run = mock.Mock(return_value=(0, ""))
        self.installer.setup_venv("app/venv", "app/requirements.txt", "fastapi", "/python")
        commands = [[str(part) for part in call.args[0]] for call in self.installer.run.call_args_list]
        self.assertFalse(any("--force-reinstall" in command for command in commands))

    def test_conflicting_flags_are_rejected(self):
        for argv in (["--yes", "--rebuild"], ["--check", "--rebuild"], ["--check", "--yes"]):
            with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    setup.arguments(argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
