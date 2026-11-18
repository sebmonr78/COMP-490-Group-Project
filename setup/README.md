# Server dependency setup

From the repository root in Ubuntu/Debian Linux x86_64, or Ubuntu on WSL2:

```bash
bash setup_server.sh
```

This installs the dependencies for the website and every simulation the website
can launch. It does not create, seed, reset or migrate a database; start a server
or simulation; change accounts or saved scans; or delete datasets and model
caches. HSSD and an initialized database remain application inputs. Existing
project source and comments are left alone by the installer.

## What it checks and installs

| Component | Dependency location and checks |
|---|---|
| Linux libraries and viewer | Build tools, OpenGL/EGL, Git/LFS, Xvfb, x11vnc, noVNC and websockify; installs missing packages with apt/sudo |
| Conda | Reuses a working installation; otherwise installs a SHA-256 verified, pinned Miniforge under `.server/conda/` |
| Habitat | The `habitat` Conda environment, Python 3.9, Habitat packages 0.3.3, Bullet support, pygame, psutil, OpenCV and Hugging Face Hub; checks imports |
| Brain | The `brain` Conda environment, Python 3.10, the existing pinned CUDA 12.8 PyTorch/xFormers/Depth Anything stack, Ultralytics 8.4.177, Piper 1.8.0, OpenCV and psutil; checks an actual CUDA calculation |
| RTAB-Map | The `rtabmap` Conda environment, RTAB-Map 0.22.1 and build tools; compiles and checks the bridge |
| Website | `app/venv/`, Python 3.10+, exact requirements and imports |
| Physics twin | `physical-drone/venv/`, Python 3.10+, requirements, imports and a MuJoCo physics step |

Package versions come from the existing requirements files, Habitat configuration
and brain setup pins. Compatible installations are reused. The website, physics
twin, Habitat and brain Python environments also run `pip check`, so matching
version numbers alone are not enough to report success. RTAB-Map is checked as
a native C++ dependency, using its Conda library/tool records, native tool
commands, and the compiled bridge's `--version` command. The brain's Python
packages follow `simulation/setup/setup_brain.sh` and the supplied setup
reference, including Ultralytics and Piper. The installer does not add an LLM
service or download model weights.

The setup reference also describes database resets, HSSD downloads, firewall
rules and shell-profile changes. Those operations remain outside this
dependency-only installer. Existing Miniconda installations are reused, without
changing their channel terms or licenses; a missing Conda installation uses
the existing verified Miniforge bootstrap instead of silently installing
Miniconda. `CONDA_ROOT` can select an existing Conda root. `HABITAT_DATA_PATH`
selects an existing dataset, while `HABITAT_DATA_ROOT` supplies the parent of
`versioned_data/hssd-hab` when no explicit dataset path was provided.

The reference marks habitat-hitl as optional, but this repository's
`simulation/config/project.json` still requires version 0.3.3 in its readiness
check. Setup therefore retains that package at 0.3.3. No application
configuration or original setup script is changed to hide that difference.

Stop the web server and simulations before repairing or rebuilding their
environments. Deactivate any environment you plan to rebuild.

The pinned CUDA build requires an NVIDIA GPU with driver 570 or newer. Setup
checks the driver before the large package installs. It does not change kernel
or GPU drivers. On WSL2, install/update the NVIDIA driver on Windows.

Habitat-Sim 0.3.3 requires Pillow 10.4.0. The Habitat package installation
explicitly includes the simulator in dependency resolution and writes
`.server/habitat-constraints.txt`, pinning Pillow to 10.4.0 and NumPy below 2.
This prevents a Habitat-Lab dependency from upgrading Pillow past the simulator's
requirement. Existing compatible dependencies are retained during this step;
it does not force-reinstall PyTorch and its CUDA libraries. Editable Habitat
packages use setuptools' supported `editable_mode=compat` build setting.
The brain and virtual-environment repair steps also keep compatible installed
dependencies. The brain selects the exact `+cu128` PyTorch/torchvision builds,
so an incompatible CUDA or CPU build is replaced without blanket reinstalls.

The Habitat window packages are installed in the Habitat environment, separately
from the brain's packages. A working NumPy 1.x version is preserved exactly
during this repair so an OpenCV update cannot replace it with NumPy 2. A working
Habitat-Lab installation is reused without cloning or replacing it. Missing pip
packages in Habitat or the brain do not trigger another Conda solve when their
Python/native core is already compatible.

If an older installer stopped with `habitat-sim 0.3.3 requires pillow==10.4.0`,
replace `setup/server_dependencies.py` with the corrected version and rerun
`bash setup_server.sh --yes`. Repair is sufficient for this version conflict;
a clean rebuild is not needed. The final version, import and `pip check`
validation still runs before setup proceeds.

RTAB-Map uses the original dependency list from `simulation/setup/setup_rtabmap.sh`:
`ros-humble-rtabmap=0.22.1`, `cxx-compiler`, `cmake`, `make` and `pkg-config`.
The project communicates with it through a compiled C++ bridge; it does not use
the RTAB-Map environment's Python interpreter or GTSAM's Python bindings. This
step therefore does not add Python, pip, NumPy constraints, or a general Python
package check to that native environment. Existing Python packages remain in
place; they are not uninstalled to make the installer pass. Native library and
bridge failures still stop setup.

Earlier installer versions incorrectly required Python-package consistency in
the native RTAB-Map environment and then added a NumPy 1.x constraint. That
constraint cannot be solved if the environment is held to Python 3.14. The
corrected installer removes the unnecessary request and does not change or
bypass Python pins. Replace `setup/server_dependencies.py` and rerun
`bash setup_server.sh --yes`; an already working native environment is reused.

The automated tests exercise installer decisions, native-command error paths,
and dependency metadata with temporary stand-ins. They do not substitute for
an actual Conda solve or the full native/GPU installation on the target server.

## Existing or incompatible installations

When an existing environment fails its checks, setup offers:

- `repair` (default): correct/install its dependency packages, then check again.
- `rebuild`: remove only that named dependency environment and recreate it.
- `cancel`: stop before modifying that environment.

Every rebuild requires an explicit interactive choice showing the affected
location. Rebuilding a virtual environment never deletes its parent directory.
The installer refuses to delete unrecognized folders, symlinked virtual
environments or an active Conda environment. Deactivate an environment before
rebuilding it. Conda's base environment is never removed.

```bash
# Inspect only. No installs, rebuilds, prompts, logs or runtime settings are written.
bash setup_server.sh --check

# Repair dependencies without prompts; this never approves deletion/rebuilding.
bash setup_server.sh --yes

# Offer a clean rebuild for each existing dependency environment, even if healthy.
# Each still needs interactive confirmation; --yes cannot be combined with this.
bash setup_server.sh --rebuild

# Point at an existing HSSD dataset without downloading or changing it.
bash setup_server.sh --data-root ~/habitat-data/versioned_data/hssd-hab
```

A failed command stops setup with its exit status. A successful command is
followed by a functional check; a failed check stops setup too. Install logs go
to `.server/logs/`. Correct the reported problem and rerun the same command;
healthy components are skipped. Only one installing process is allowed per
project at a time. `--check` returns a nonzero exit status if a dependency is
missing or incompatible.

## Start the server

After setup succeeds, run from the repository root:

```bash
app/venv/bin/python -m uvicorn main:app --app-dir app --host 0.0.0.0 --port 8000
```

You can also activate `app/venv` and use the existing Uvicorn command from
`app/README.md`. The app reads the generated `.server/environment.json` to find
Conda and the existing HSSD path, so manual Conda activation is unnecessary.
Explicit environment variables such as `CONDA_EXE` and `HABITAT_DATA_PATH`
override those defaults. WSL-specific rendering defaults are applied only on
WSL and do not replace explicit renderer settings. No shell profile is changed.

The installer does not download HSSD, fetch model weights, accept account terms,
create demo accounts, expose firewall ports or launch services. The existing
dataset instructions remain in `simulation/README.md`; Depth Anything can fetch
its weights through its existing model loader when used. These inputs are
preserved across repairs and rebuilds.

## Map saving after a run

The website keeps a DELTA run's slot until its map has been imported. Its final
Output line is one of:

```text
[website] Map saved to your account as scan N.
[website] No new map to save.
[website] Could not save the map to your account: REASON
```

An unchanged map left by a previous run is not imported. The run remains busy
while saving; after a successful save the dashboard refreshes and selects the
new scan automatically. The launcher waits for the brain to finish before the
website imports the map, within the existing 45-second stop allowance.

The latest map importer is retained: it locates maps inside
`memory/<scene>/<pose source>/` and skips folders ending in `.previous`.
The unchanged-map check uses the same folder finder and records all existing
maps before starting, so reusing a slot does not save another user's old map.

## Checks for changes to the installer and website

```bash
python3 -m unittest discover -s setup -p 'test_*.py'
app/venv/bin/python app/test_server_environment.py
app/venv/bin/python app/test_app.py
python3 database/test_import_run.py
python3 database/test_db.py
node app/test_dashboard.js
```

Node is only needed for the JavaScript polling test; it is not a server runtime
dependency. These tests use temporary databases and folders and do not install
system packages or contact package repositories. A full installation and GPU
rendering test must run on the intended NVIDIA-equipped server.
