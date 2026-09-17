import importlib.metadata
import json
import os
import platform
import subprocess
import warnings
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "project.json"

# Filled in from project.json by main(), so the banner and the settings it
# reports can never disagree about what this project is called.
TITLE = "DELTA"


def add_result(results, name, status, details):
    results.append(
        {
            "name": name,
            "status": status,
            "details": details,
        }
    )


def check_package(results, package_name, required_version):
    try:
        installed_version = importlib.metadata.version(package_name)

        if installed_version == required_version:
            add_result(
                results,
                package_name,
                "PASS",
                f"version {installed_version}",
            )
        else:
            add_result(
                results,
                package_name,
                "FAIL",
                f"required {required_version}, found {installed_version}",
            )
    except importlib.metadata.PackageNotFoundError:
        add_result(
            results,
            package_name,
            "FAIL",
            "not installed",
        )


def check_file(results, name, path, required=True):
    if path.is_file():
        add_result(results, name, "PASS", str(path))
    elif required:
        add_result(results, name, "FAIL", f"missing: {path}")
    else:
        add_result(results, name, "WARN", f"not created: {path}")


def check_nvidia(results):
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if process.returncode == 0:
            output = process.stdout.strip()
            add_result(results, "NVIDIA GPU", "PASS", output)
        else:
            add_result(
                results,
                "NVIDIA GPU",
                "FAIL",
                process.stderr.strip() or "nvidia-smi failed",
            )
    except FileNotFoundError:
        add_result(results, "NVIDIA GPU", "FAIL", "nvidia-smi not found")
    except Exception as error:
        add_result(results, "NVIDIA GPU", "FAIL", str(error))


def print_summary(results):
    print()
    print("=" * 72)
    print(f"{TITLE} - ENVIRONMENT CHECK")
    print("=" * 72)

    for result in results:
        print(
            f"{result['status']:<5} "
            f"{result['name']:<25} "
            f"{result['details']}"
        )

    failures = [
        result for result in results if result["status"] == "FAIL"
    ]

    warning_rows = [
        result for result in results if result["status"] == "WARN"
    ]

    print("=" * 72)

    if failures:
        print(f"SETUP STATUS: NOT READY ({len(failures)} failure(s))")
        return 1

    if warning_rows:
        print(
            f"SETUP STATUS: READY WITH "
            f"{len(warning_rows)} WARNING(S)"
        )
        return 0

    print("SETUP STATUS: READY")
    return 0


def main():
    results = []

    if not CONFIG_PATH.is_file():
        print(f"Missing project configuration: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    global TITLE
    TITLE = config.get("project_name", TITLE)

    add_result(
        results,
        "Operating system",
        "PASS",
        f"{platform.system()} {platform.machine()}",
    )

    active_environment = os.environ.get(
        "CONDA_DEFAULT_ENV",
        "not active",
    )

    required_environment = config["required_conda_environment"]

    if active_environment == required_environment:
        add_result(
            results,
            "Conda environment",
            "PASS",
            active_environment,
        )
    else:
        add_result(
            results,
            "Conda environment",
            "FAIL",
            f"required {required_environment}, "
            f"active {active_environment}",
        )

    installed_python = platform.python_version()
    required_python = config["required_python"]

    # Habitat asks for Python 3.9; it does not care which bugfix release.
    # Compare piece by piece so "3.9" accepts 3.9.25 and 3.9.26 alike, while
    # a plain string compare would reject everything except an exact match.
    required_parts = required_python.split(".")
    installed_parts = installed_python.split(".")

    if installed_parts[: len(required_parts)] == required_parts:
        add_result(
            results,
            "Python",
            "PASS",
            installed_python,
        )
    else:
        add_result(
            results,
            "Python",
            "FAIL",
            f"required {required_python}, found {installed_python}",
        )

    for package_name, version in config[
        "required_packages"
    ].items():
        check_package(results, package_name, version)

    try:
        import habitat_sim
        import habitat

        add_result(
            results,
            "Habitat imports",
            "PASS",
            "habitat_sim and habitat imported",
        )
    except Exception as error:
        add_result(
            results,
            "Habitat imports",
            "FAIL",
            f"{type(error).__name__}: {error}",
        )

    check_nvidia(results)

    dataset_config = config["dataset"]
    environment_variable = dataset_config["environment_variable"]

    dataset_root_value = os.environ.get(
        environment_variable,
        dataset_config["default_root"],
    )

    dataset_root = Path(dataset_root_value).expanduser().resolve()

    add_result(
        results,
        "HSSD root",
        "PASS" if dataset_root.is_dir() else "FAIL",
        str(dataset_root),
    )

    dataset_config_path = (
        dataset_root / dataset_config["configuration_file"]
    )
    scene_path = dataset_root / dataset_config["scene_file"]
    semantic_path = dataset_root / dataset_config["semantic_file"]
    stage_path = dataset_root / dataset_config["stage_file"]

    check_file(
        results,
        "HSSD configuration",
        dataset_config_path,
    )
    check_file(results, "Selected scene", scene_path)
    check_file(results, "Semantic configuration", semantic_path)
    check_file(results, "Stage model", stage_path)

    # DELTA does not use HSSD's own navmesh, and does not check for one.
    # HSSD's describes where a walking human fits; ours describes where a
    # 30 cm ball fits, which is a different map of the same building -- see
    # section 12 of the README. open_environment.py builds ours on first run
    # and saves it, so not having one yet is normal and only worth a warning.
    check_file(
        results,
        "Drone navigation mesh",
        dataset_root / dataset_config["drone_navmesh_file"],
        required=False,
    )

    # Rendering is deliberately NOT tested here. Creating a Simulator on WSL
    # aborts inside the C++ layer when EGL cannot find a CUDA device, and a
    # C++ abort kills the process outright -- a Python try/except cannot catch
    # it, so this script would die before printing any summary at all.
    # open_environment.py verifies real rendering instead.

    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())