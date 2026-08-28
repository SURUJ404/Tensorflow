"""Build all chapter notebooks from the cell definitions in tools/builders."""
import sys, importlib, pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

BUILDERS = ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06",
            "ch07", "ch08", "ch09", "ch10", "ch11", "ch12"]


def main() -> None:
    reports = []
    for name in BUILDERS:
        mod = importlib.import_module(f"builders.{name}")
        tag = getattr(mod, "build")()
        reports.append(f"built {tag}")
    print("\n".join(reports))
    print("All notebooks built.")


if __name__ == "__main__":
    main()