"""Small, non-destructive safeguards shared by the public training scripts."""

from pathlib import Path


def prepare_empty_run_dirs(*paths):
    """Create run directories only when none already contains artifacts.

    Trainers do not implement checkpoint resume. Reusing a non-empty directory
    could mix checkpoints from different epoch budgets and make last-K select an
    older high-epoch file. Fail loudly instead of deleting or overwriting data.
    """
    run_dirs = [Path(path) for path in paths]
    for run_dir in run_dirs:
        if run_dir.exists():
            if not run_dir.is_dir():
                raise FileExistsError(f"Run output path exists but is not a directory: {run_dir}")
            if next(run_dir.iterdir(), None) is not None:
                raise FileExistsError(
                    "Refusing to reuse a non-empty run directory because stale "
                    f"checkpoints could contaminate last-K selection: {run_dir}\n"
                    "Move or archive that directory, or choose a different method/seed/fold/configuration."
                )
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True, exist_ok=True)
