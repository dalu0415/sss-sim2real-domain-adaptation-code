"""Serial25-run orchestration. Default is a printed plan, --execute explicitly runs it."""
import argparse
import subprocess
import sys
from pathlib import Path

from common import check_environment, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate", "aggregate", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--authorize-klsg-labels", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args()
    check_environment()
    if args.execute and args.stage in ("all", "evaluate"):
        require(args.authorize_klsg_labels and bool(args.reason and args.reason.strip()), "Explicit final scoring authorization/reason required")
    commands = []
    directory = Path(__file__).resolve().parent
    for seed in range(5):
        for fold in range(5):
            for stage, script in (("train", "train_asda.py"), ("evaluate", "evaluate_asda.py")):
                if args.stage not in (stage, "all"):
                    continue
                cmd = [sys.executable, "-X", "utf8", str(directory / script), "--seed", str(seed), "--fold", str(fold)]
                if args.restart_incomplete:
                    cmd.append("--restart-incomplete")
                if stage == "evaluate":
                    cmd += ["--authorize-klsg-labels", "--reason", args.reason or "FINAL SCORING REASON REQUIRED"]
                commands.append(cmd)
    if args.stage in ("all", "aggregate"):
        commands.append([sys.executable, "-X", "utf8", str(directory / "evaluate_asda.py"), "--aggregate"])
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
    print("EXECUTED" if args.execute else "PLAN ONLY: no model training or target scoring executed")


if __name__ == "__main__":
    main()
