"""Install, inspect and repair DELTA's server dependency environments.

This installer manages dependencies only. Database creation, seeding, running
simulations and starting the web server remain separate operations.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MINIFORGE_VERSION = "26.9.4-2"
MINIFORGE_SHA256 = "bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf"
MINIFORGE_URL = (
    "https://github.com/conda-forge/miniforge/releases/download/"
    + MINIFORGE_VERSION + "/Miniforge3-" + MINIFORGE_VERSION + "-Linux-x86_64.sh"
)
SYSTEM_PACKAGES = (
    "ca-certificates", "curl", "wget", "git", "git-lfs", "build-essential", "pkg-config", "psmisc",
    "python3-venv", "xvfb", "xauth", "x11vnc", "novnc", "websockify",
    "libegl1", "libgl1", "libgl1-mesa-dri", "libopengl0", "libglfw3",
    "libglu1-mesa", "libxrandr2", "libxi6", "libxcursor1", "libxinerama1",
)
ENV_NAMES = ("habitat", "brain", "rtabmap")
HABITAT_REQUIREMENTS = ("habitat-sim==0.3.3", "Pillow==10.4.0", "numpy<2", "huggingface_hub<1")
HABITAT_WINDOW_REQUIREMENTS = ("pygame", "psutil", "opencv-python", "huggingface_hub<1")
RTABMAP_REQUIREMENTS = ("ros-humble-rtabmap=0.22.1", "cxx-compiler", "cmake", "make", "pkg-config")

REQUIREMENT_PROBE = r'''
import importlib, importlib.metadata as metadata, sys
from pip._vendor.packaging.requirements import Requirement
assert sys.version_info >= (3, 10), "Python 3.10+ is required"
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    required = Requirement(line)
    actual = metadata.version(required.name)
    assert actual in required.specifier, "%s: found %s, need %s" % (required.name, actual, required.specifier)
for name in sys.argv[2].split(","):
    importlib.import_module(name)
if "mujoco" in sys.argv[2]:
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    mujoco.mj_step(model, mujoco.MjData(model))
print("Required versions and imports work")
'''

HABITAT_CORE_PROBE = r'''
import importlib.metadata as metadata, sys, pip
assert sys.version_info[:2] == (3, 9), "Habitat needs Python 3.9"
assert metadata.version("habitat-sim") == "0.3.3", "habitat-sim must be 0.3.3"
import habitat_sim
assert habitat_sim.built_with_bullet, "Habitat was built without Bullet physics"
'''

HABITAT_LAB_PROBE = r'''
import importlib.metadata as metadata
for name in ("habitat-lab", "habitat-baselines", "habitat-hitl"):
    assert metadata.version(name) == "0.3.3", name + " must be 0.3.3"
import habitat, habitat_baselines, habitat_hitl
'''

BRAIN_CORE_PROBE = r'''
import sys, pip
assert sys.version_info[:2] == (3, 10), "The brain needs Python 3.10"
'''

HABITAT_PROBE = r'''
import importlib.metadata as metadata, sys
assert sys.version_info[:2] == (3, 9), "Habitat needs Python 3.9"
for name in ("habitat-sim", "habitat-lab", "habitat-baselines", "habitat-hitl"):
    assert metadata.version(name) == "0.3.3", name + " must be 0.3.3"
import habitat_sim, habitat, habitat_baselines, habitat_hitl, magnum, cv2, huggingface_hub, pygame, psutil
assert habitat_sim.built_with_bullet, "Habitat was built without Bullet physics"
print("Habitat 0.3.3, Bullet, Magnum and window packages import successfully")
'''

BRAIN_PROBE = r'''
import importlib.metadata as metadata, json, sys
assert sys.version_info[:2] == (3, 10), "The brain needs Python 3.10"
pins = json.loads(sys.argv[1])
for name, key in (("torch", "TORCH"), ("torchvision", "TORCHVISION"), ("xformers", "XFORMERS"),
                  ("ultralytics", "ULTRALYTICS"), ("piper-tts", "PIPER")):
    assert metadata.version(name).split("+")[0] == pins[key], name + " has the wrong version"
info = json.loads(metadata.distribution("depth-anything-3").read_text("direct_url.json") or "{}")
assert info.get("vcs_info", {}).get("commit_id") == pins["DA3_COMMIT"], "Depth Anything source revision differs"
import cv2, numpy, psutil, torch, torchvision, xformers.ops, depth_anything_3.api, ultralytics, piper
assert torch.version.cuda == "12.8", "PyTorch must use the CUDA 12.8 build"
assert torch.cuda.is_available(), "PyTorch cannot use the NVIDIA GPU"
assert float((torch.ones(1, device="cuda") + 1).item()) == 2
print("Brain versions, imports and a CUDA calculation work")
'''

class SetupError(Exception):
    pass


def read_pins(root):
    """Reuse the existing brain setup's pinned stack instead of upgrading it."""
    source = (root / "simulation/setup/setup_brain.sh").read_text()
    pins = dict(re.findall(r'^([A-Z][A-Z0-9_]*)="([^"\n]*)"', source, re.M))
    required = ("TORCH", "TORCHVISION", "TORCH_INDEX", "XFORMERS", "DA3_COMMIT", "ULTRALYTICS", "PIPER")
    if any(not pins.get(name) for name in required):
        raise SetupError("The brain setup is missing a required dependency version pin.")
    return pins


class Installer:
    def __init__(self, root, args, ask=input):
        self.root = Path(root).resolve()
        self.args = args
        self.ask = ask
        self.state = self.root / ".server"
        self.conda = None
        self.log = None
        self.failures = []
        self.pins = read_pins(self.root)
        self.env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                        PIP_DISABLE_PIP_VERSION_CHECK="1", MPLBACKEND="Agg")
        self.env.pop("PYTHONPATH", None)
        self.env.pop("PYTHONHOME", None)
        virtual_env = self.env.pop("VIRTUAL_ENV", None)
        if virtual_env:
            folders = self.env.get("PATH", "").split(os.pathsep)
            self.env["PATH"] = os.pathsep.join(folder for folder in folders
                                              if Path(folder).resolve() != Path(virtual_env).resolve() / "bin")

    def say(self, message):
        print(message, flush=True)
        if self.log:
            self.log.write(message + "\n")
            self.log.flush()

    def run(self, command, capture=False, timeout=None, env=None):
        command = [str(part) for part in command]
        if not capture:
            self.say("$ " + shlex.join(command))
        try:
            if capture:
                result = subprocess.run(command, cwd=self.root, env=env or self.env,
                                        text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                return result.returncode, result.stdout.strip()
            with subprocess.Popen(command, cwd=self.root, env=env or self.env,
                                  text=True, errors="replace", stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
                for line in process.stdout:
                    self.say(line.rstrip("\n"))
                code = process.wait()
            if code:
                raise SetupError("Command failed with exit status %s: %s" % (code, shlex.join(command)))
            return 0, ""
        except (OSError, subprocess.TimeoutExpired) as error:
            if capture:
                return 1, str(error)
            raise SetupError(str(error)) from error

    def problem(self, label, detail):
        self.say("[missing/wrong] %s: %s" % (label, detail))
        self.failures.append(label)

    def check_os(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise SetupError("Use Linux x86_64, or Ubuntu on WSL2.")
        if not shutil.which("apt-get") or not shutil.which("dpkg-query"):
            raise SetupError("Automatic system-package installation needs Ubuntu/Debian and apt-get.")
        self.say("[ok] Linux x86_64 (Ubuntu/Debian, including Ubuntu on WSL2)")

    def check_gpu(self):
        code, output = self.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                 "--format=csv,noheader"], capture=True, timeout=20)
        versions = re.findall(r",\s*(\d+)\.\d+", output) if not code else []
        if not versions or any(int(version) < 570 for version in versions):
            detail = "The pinned CUDA stack needs a working NVIDIA driver 570+. "
            detail += "On WSL2, update the Windows NVIDIA driver; do not install a Linux GPU driver in WSL."
            if self.args.check:
                self.problem("NVIDIA driver", detail)
                return
            raise SetupError(detail)
        self.say("[ok] NVIDIA GPU: " + output.replace("\n", "; "))

    def system_packages(self):
        missing = []
        for name in SYSTEM_PACKAGES:
            code, output = self.run(["dpkg-query", "-W", "-f=${Status}", name], capture=True)
            if code or output != "install ok installed":
                missing.append(name)
        broken = [name for name in ("git", "git-lfs", "Xvfb", "x11vnc", "websockify")
                  if not shutil.which(name)]
        if not missing and not broken and Path("/usr/share/novnc/vnc.html").is_file():
            self.say("[ok] System libraries, build tools and browser-viewer programs")
            return
        if self.args.check:
            self.problem("System packages", ", ".join(missing or broken or ["noVNC web files"]))
            return
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        if sudo and not shutil.which("sudo"):
            raise SetupError("System dependencies need sudo or an administrator.")
        self.run(sudo + ["apt-get", "update"])
        providers = {"git": "git", "git-lfs": "git-lfs", "Xvfb": "xvfb",
                     "x11vnc": "x11vnc", "websockify": "websockify"}
        needed = set(missing) | {providers[name] for name in broken}
        if not Path("/usr/share/novnc/vnc.html").is_file():
            needed.add("novnc")
        self.run(sudo + ["apt-get", "install", "-y", "--reinstall"] + sorted(needed))
        for name in ("git", "git-lfs", "Xvfb", "x11vnc", "websockify"):
            if not shutil.which(name):
                raise SetupError("System package installation did not provide " + name)
        if not Path("/usr/share/novnc/vnc.html").is_file():
            raise SetupError("The noVNC web files are still missing.")

    def find_conda(self):
        configured_root = os.environ.get("CONDA_ROOT")
        configured = str(Path(configured_root).expanduser() / "bin/conda") if configured_root else None
        candidates = [configured, os.environ.get("CONDA_EXE"), shutil.which("conda"),
                      str(self.state / "conda/bin/conda"),
                      str(Path.home() / "miniconda3/bin/conda"),
                      str(Path.home() / "miniforge3/bin/conda")]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                code, output = self.run([candidate, "--version"], capture=True, timeout=30)
                if not code and output.startswith("conda "):
                    return str(Path(candidate).resolve())
        return None

    def setup_conda(self):
        self.conda = self.find_conda()
        if not self.conda:
            if self.args.check:
                self.problem("Conda", "No working Conda installation was found")
                return
            destination = self.state / "conda"
            if destination.exists():
                if destination.is_symlink():
                    raise SetupError("Refusing to replace a symlinked Conda installation: " + str(destination))
                self.choice("Incomplete project Conda installation", destination, can_repair=False)
                backup = destination.with_name("conda-incomplete-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
                destination.rename(backup)
                self.say("The incomplete dependency installation was moved to " + str(backup))
            self.say("Installing a private Miniforge distribution (SHA-256 verified).")
            with tempfile.TemporaryDirectory(prefix="delta-conda-") as temp:
                installer = Path(temp) / "miniforge.sh"
                try:
                    with urllib.request.urlopen(MINIFORGE_URL, timeout=60) as response, installer.open("wb") as output:
                        shutil.copyfileobj(response, output)
                except OSError as error:
                    raise SetupError("Could not download Miniforge; rerun to retry: " + str(error)) from error
                digest = hashlib.sha256()
                with installer.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != MINIFORGE_SHA256:
                    raise SetupError("Miniforge checksum verification failed; the installer was not executed.")
                self.run(["bash", installer, "-b", "-p", destination])
            self.conda = str(destination / "bin/conda")
        self.env["PATH"] = str(Path(self.conda).parent) + os.pathsep + self.env.get("PATH", "")
        self.env["CONDA_EXE"] = self.conda
        self.say("[ok] Conda: " + self.conda)

    def conda_prefix(self, name):
        if not self.conda:
            return None
        code, output = self.run([self.conda, "info", "--json"], capture=True, timeout=60)
        if code:
            raise SetupError("Could not inspect Conda: " + output)
        try:
            info = json.loads(output)
            for directory in info["envs_dirs"]:
                candidate = Path(directory) / name
                if candidate.is_dir():
                    return candidate
            return Path(info["envs_dirs"][0]) / name
        except (KeyError, ValueError, IndexError) as error:
            raise SetupError("Conda returned invalid environment information") from error

    def choice(self, label, path, can_repair=True):
        if self.args.rebuild:
            prompt = "%s at %s: type rebuild to replace this dependency environment, or cancel: " % (label, path)
        elif self.args.yes:
            if not can_repair:
                raise SetupError("%s has an incompatible interpreter; rerun interactively and choose rebuild." % label)
            return "repair"
        else:
            prompt = "%s at %s: %srebuild, or cancel? " % (label, path, "repair [default], " if can_repair else "")
        if not sys.stdin.isatty() and self.ask is input:
            raise SetupError("An existing environment needs a decision. Run interactively, or use --yes for repairs only.")
        answer = self.ask(prompt).strip().lower()
        if not answer and can_repair and not self.args.rebuild:
            return "repair"
        if answer in ("repair", "rebuild") and (answer != "repair" or can_repair) and not (self.args.rebuild and answer != "rebuild"):
            return answer
        raise SetupError("Cancelled; no rebuild was performed for " + label)

    def ensure(self, label, path, probe, repair, rebuild, can_repair=True):
        ok, detail = probe()
        if ok and not self.args.rebuild:
            self.say("[ok] %s: %s" % (label, detail))
            return
        if self.args.check:
            if ok:
                self.say("[ok] %s: %s" % (label, detail))
            else:
                self.problem(label, detail)
            return
        exists = path is not None and path.exists()
        action = self.choice(label, path, can_repair) if exists else "repair"
        self.say("[%s] %s: %s" % (action, label, detail))
        try:
            (rebuild if action == "rebuild" else repair)()
        except SetupError:
            self.say("The dependency step failed. Rerun setup to retry, repair, or choose a clean rebuild.")
            raise
        ok, detail = probe()
        if not ok:
            raise SetupError("%s still failed validation: %s. Rerun and choose rebuild if repair did not resolve it." % (label, detail))
        self.say("[ok] %s validated" % label)

    def python_probe(self, python, source, *arguments):
        if python is None or not Path(python).is_file():
            return False, "Python/environment is missing"
        code, detail = self.run([python, "-c", source] + list(arguments), capture=True, timeout=180)
        if code:
            return False, detail[-4000:]
        code, conflict = self.run([python, "-m", "pip", "check"], capture=True, timeout=60)
        return (False, conflict) if code else (True, detail)

    def interpreter_matches(self, python, version):
        code, _ = self.run([python, "-c", "import sys; assert sys.version_info[:2] == " + repr(version)], capture=True)
        return not code

    def conda_python_probe(self, prefix, source):
        python = prefix / "bin/python"
        if not python.is_file():
            return False, "Python/environment is missing"
        code, detail = self.run([python, "-c", source], capture=True, timeout=180)
        return not code, detail

    def remove_conda_env(self, name, prefix):
        if name not in ENV_NAMES or prefix is None or prefix.is_symlink() or prefix.name != name or not (prefix / "conda-meta").is_dir():
            raise SetupError("Refusing to remove an unrecognized dependency environment: " + str(prefix))
        if os.environ.get("CONDA_PREFIX") and prefix.resolve() == Path(os.environ["CONDA_PREFIX"]).resolve():
            raise SetupError("Deactivate %s before rebuilding it, then rerun setup." % name)
        self.run([self.conda, "env", "remove", "--prefix", prefix, "--yes"])

    def conda_command(self, prefix, *command):
        return [self.conda, "run", "--no-capture-output", "--prefix", prefix] + list(command)

    def setup_conda_env(self, name, specs, probe_source, probe_args=(), after=None, check=None, core_check=None):
        prefix = self.conda_prefix(name)
        if prefix is None:
            self.problem(name, "Conda is not available")
            return None
        python = prefix / "bin/python"
        channels = ["robostack-staging", "conda-forge"] if name == "rtabmap" else ["conda-forge", "aihabitat"]
        channel_args = [part for channel in channels for part in ("-c", channel)]

        def probe():
            return check(prefix) if check is not None else self.python_probe(python, probe_source, *probe_args)

        def repair():
            command = "install" if (prefix / "conda-meta").is_dir() else "create"
            core_ready = command == "install" and core_check is not None and core_check(prefix)[0]
            if not core_ready:
                self.run([self.conda, command, "--prefix", prefix, "--yes", "--override-channels"] + channel_args + list(specs))
            if after:
                after(prefix)

        def rebuild():
            self.remove_conda_env(name, prefix)
            repair()

        self.ensure(name, prefix, probe, repair, rebuild)
        return prefix

    def rtabmap_probe(self, prefix):
        if not (prefix / "conda-meta").is_dir():
            return False, "RTAB-Map dependency environment is missing"
        code, output = self.run([self.conda, "list", "--prefix", prefix, "--json"], capture=True, timeout=60)
        if code:
            return False, "Could not inspect RTAB-Map native packages: " + output
        try:
            records = {record["name"]: record for record in json.loads(output)}
        except (ValueError, TypeError, KeyError):
            return False, "Conda returned invalid RTAB-Map package information"
        if records.get("ros-humble-rtabmap", {}).get("version") != "0.22.1":
            return False, "RTAB-Map native library 0.22.1 is required"
        missing = [name for name in ("cxx-compiler", "cmake", "make", "pkg-config") if name not in records]
        if missing:
            return False, "Missing native build dependencies: " + ", ".join(missing)
        for program in ("cmake", "make", "pkg-config"):
            code, output = self.run(self.conda_command(prefix, program, "--version"), capture=True, timeout=60)
            if code:
                return False, "RTAB-Map native tool %s failed: %s" % (program, output)
        return True, "RTAB-Map 0.22.1 native library and build tools; C++ bridge is checked separately"

    def habitat_packages(self, prefix):
        source = self.state / "sources/habitat-lab"
        editable = []
        if not self.conda_python_probe(prefix, HABITAT_LAB_PROBE)[0]:
            if source.exists():
                code, tag = self.run(["git", "-C", source, "describe", "--tags", "--exact-match"], capture=True)
                clean, dirty = self.run(["git", "-C", source, "status", "--porcelain"], capture=True)
                if code or tag != "v0.3.3" or clean or dirty:
                    raise SetupError("The dependency checkout %s is modified or has the wrong tag; move it aside before retrying." % source)
            else:
                source.parent.mkdir(parents=True, exist_ok=True)
                self.run(["git", "clone", "--depth", "1", "--branch", "v0.3.3",
                          "https://github.com/facebookresearch/habitat-lab.git", source])
            editable = ["-e", source / "habitat-lab", "-e", source / "habitat-baselines",
                        "-e", source / "habitat-hitl"]
        self.state.mkdir(parents=True, exist_ok=True)
        constraints = self.state / "habitat-constraints.txt"
        requirements = list(HABITAT_REQUIREMENTS)
        code, numpy_version = self.run([prefix / "bin/python", "-c", "import numpy; print(numpy.__version__)"],
                                       capture=True, timeout=60)
        if not code and re.fullmatch(r"1\.\d+\.\d+(?:[A-Za-z0-9.+-]*)", numpy_version):
            requirements.append("numpy==" + numpy_version)
        constraints.write_text("\n".join(requirements) + "\n")
        self.run([prefix / "bin/python", "-m", "pip", "install", "--constraint", constraints,
                  "--upgrade-strategy", "only-if-needed", "--config-settings", "editable_mode=compat"] +
                 requirements + list(HABITAT_WINDOW_REQUIREMENTS) + editable)

    def brain_packages(self, prefix):
        python = prefix / "bin/python"
        packages = ["torch==" + self.pins["TORCH"] + "+cu128", "torchvision==" + self.pins["TORCHVISION"] + "+cu128",
                    "xformers==" + self.pins["XFORMERS"]]
        self.run([python, "-m", "pip", "install", "--index-url", self.pins["TORCH_INDEX"]] + packages)
        self.state.mkdir(parents=True, exist_ok=True)
        constraints = self.state / "brain-constraints.txt"
        constraints.write_text("\n".join(packages) + "\n")
        self.run([python, "-m", "pip", "install", "--constraint", constraints,
                  "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3@" + self.pins["DA3_COMMIT"],
                  "ultralytics==" + self.pins["ULTRALYTICS"], "piper-tts==" + self.pins["PIPER"],
                  "addict", "opencv-python", "numpy", "psutil"])

    def setup_venv(self, relative, requirements, imports, base_python):
        path = self.root / relative
        python = path / "bin/python"
        label = relative + " dependencies"

        def probe():
            return self.python_probe(python, REQUIREMENT_PROBE, str(self.root / requirements), imports)

        def repair():
            if not python.is_file():
                if path.exists():
                    raise SetupError("%s is incomplete; choose rebuild to replace it." % path)
                self.run([base_python, "-m", "venv", path])
            pip_status, _ = self.run([python, "-m", "pip", "--version"], capture=True)
            if pip_status:
                self.run([python, "-m", "ensurepip", "--upgrade"])
            self.run([python, "-m", "pip", "install", "--upgrade", "pip"])
            self.run([python, "-m", "pip", "install", "-r", self.root / requirements])

        def rebuild():
            if path.is_symlink() or path.resolve() != (self.root / relative).absolute() or not (path / "pyvenv.cfg").is_file():
                raise SetupError("Refusing to delete a folder that is not a recognized project virtual environment: " + str(path))
            active = os.environ.get("VIRTUAL_ENV")
            if active and Path(active).resolve() == path.resolve():
                raise SetupError("Deactivate %s before rebuilding it, then rerun setup." % path)
            shutil.rmtree(path)
            repair()

        code, _ = self.run([python, "-c", "import sys; assert sys.version_info >= (3, 10)"], capture=True) if python.is_file() else (1, "")
        self.ensure(label, path, probe, repair, rebuild, can_repair=not path.exists() or code == 0)

    def setup_bridge(self, prefix):
        bridge = self.root / "simulation/build/rtabmap_bridge/rtabmap_bridge"
        if prefix is None:
            self.problem("RTAB-Map bridge", "RTAB-Map environment is unavailable")
            return
        code, output = self.run(self.conda_command(prefix, bridge, "--version"), capture=True, timeout=60)
        if not code and "0.22.1" in output and not self.args.rebuild:
            self.say("[ok] RTAB-Map bridge: " + output)
            return
        if self.args.check:
            self.problem("RTAB-Map bridge", output)
            return
        build = self.state / "build/rtabmap_bridge"
        build.mkdir(parents=True, exist_ok=True)
        self.run(self.conda_command(prefix, "cmake", "-S", self.root / "simulation/delta/brain/rtabmap_bridge",
                                   "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH=" + str(prefix)))
        self.run(self.conda_command(prefix, "cmake", "--build", build, "--parallel", str(min(os.cpu_count() or 2, 4))))
        candidate = build / "rtabmap_bridge"
        code, output = self.run(self.conda_command(prefix, candidate, "--version"), capture=True, timeout=60)
        if code or "0.22.1" not in output:
            raise SetupError("The newly built RTAB-Map bridge failed: " + output)
        bridge.parent.mkdir(parents=True, exist_ok=True)
        temporary = bridge.with_suffix(".installing")
        shutil.copy2(candidate, temporary)
        temporary.replace(bridge)
        self.say("[ok] RTAB-Map bridge compiled and validated")

    def write_environment(self):
        project = json.loads((self.root / "simulation/config/project.json").read_text())
        data_root = os.environ.get("HABITAT_DATA_ROOT")
        dataset = (self.args.data_root or os.environ.get("HABITAT_DATA_PATH") or
                   (str(Path(data_root).expanduser() / "versioned_data/hssd-hab") if data_root else None) or
                   project["dataset"]["default_root"])
        config = {"conda_executable": self.conda, "dataset_root": str(Path(dataset).expanduser().resolve())}
        release = platform.release().lower()
        if "microsoft" in release or Path("/dev/dxg").exists():
            config["wsl_rendering"] = True
        destination = self.state / "environment.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(destination)
        data_root = Path(config["dataset_root"])
        if not (data_root / project["dataset"]["configuration_file"]).is_file():
            self.say("[notice] HSSD is not present at %s. Supply the existing dataset with --data-root PATH or HABITAT_DATA_PATH." % data_root)

    def execute(self):
        self.check_os()
        if not self.args.check:
            logs = self.state / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            self.log = (logs / ("setup-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")).open("a", encoding="utf-8")
            import fcntl
            self.lock = (self.state / "setup.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SetupError("Another dependency setup is running for this project.") from error
        self.check_gpu()
        self.system_packages()
        self.setup_conda()
        habitat = self.setup_conda_env("habitat", ("python=3.9", "pip", "numpy<2", "cmake=3.14.0", "habitat-sim=0.3.3", "withbullet"),
                                      HABITAT_PROBE, after=self.habitat_packages,
                                      core_check=lambda prefix: self.conda_python_probe(prefix, HABITAT_CORE_PROBE))
        brain = self.setup_conda_env("brain", ("python=3.10", "pip"), BRAIN_PROBE,
                                    (json.dumps(self.pins),), after=self.brain_packages,
                                    core_check=lambda prefix: self.conda_python_probe(prefix, BRAIN_CORE_PROBE))
        rtabmap = self.setup_conda_env("rtabmap", RTABMAP_REQUIREMENTS, None, check=self.rtabmap_probe)
        self.setup_bridge(rtabmap)
        base_python = str(brain / "bin/python") if brain else sys.executable
        self.setup_venv("app/venv", "app/requirements.txt", "fastapi,uvicorn,httpx", base_python)
        self.setup_venv("physical-drone/venv", "physical-drone/requirements.txt", "mujoco,glfw,numpy,matplotlib,PIL", base_python)
        if self.failures:
            self.say("Not ready: " + ", ".join(self.failures))
            return 1
        if self.args.check:
            self.say("All server dependency checks passed.")
            return 0
        self.write_environment()
        self.say("Server dependencies are ready. No database was created, seeded or reset.")
        self.say("Start the server from the project root:")
        self.say("  app/venv/bin/python -m uvicorn main:app --app-dir app --host 0.0.0.0 --port 8000")
        self.say("An existing initialized database and HSSD dataset are application inputs, not installed dependencies.")
        return 0


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="inspect dependencies only; install and rebuild nothing")
    parser.add_argument("--yes", action="store_true", help="repair existing environments without prompts; never approve deletion")
    parser.add_argument("--rebuild", action="store_true", help="offer a clean rebuild for each existing dependency environment; requires interactive confirmation")
    parser.add_argument("--data-root", help="location of an already downloaded HSSD dataset; no data is downloaded or removed")
    args = parser.parse_args(argv)
    if args.check and (args.rebuild or args.yes):
        parser.error("--check cannot be combined with --yes or --rebuild")
    if args.rebuild and args.yes:
        parser.error("--rebuild requires interactive confirmation; it cannot be combined with --yes")
    return args


def main(argv=None):
    if sys.version_info < (3, 8):
        print("Use Python 3.8+ to run this installer.", file=sys.stderr)
        return 1
    installer = None
    try:
        installer = Installer(ROOT, arguments(argv))
        return installer.execute()
    except (SetupError, OSError, ValueError, KeyboardInterrupt, EOFError) as error:
        report = installer.say if installer else print
        report("Setup stopped: " + (str(error) or "cancelled"))
        report("Rerun bash setup_server.sh to retry; successful dependency checks will be skipped.")
        return 1
    finally:
        if installer and installer.log:
            installer.log.close()
        if installer and getattr(installer, "lock", None):
            installer.lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
