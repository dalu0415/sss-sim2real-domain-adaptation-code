"""Fixed40-epoch ASDA trainer. Does not read target task labels or score target accuracy."""
import argparse
import datetime
import time
import traceback

from common import (check_environment, set_seed, make_config, require, run_root,
                    write_json, append_json, EPOCHS)
import torch
from data import training_loaders
from model import ASDAResNet18, Discriminator, optimizer_scheduler, train_step, bn_counts
from artifacts import checkpoint_payload, complete_training, write_training_complete


class ForwardMACs:
    """Conv2d+Linear multiply-accumulates from observed shapes (not full FLOPs)."""
    def __init__(self, model, discriminator):
        self.counts = {"G": 0, "C": 0, "D": 0}
        self.handles = []
        for tag, module in (("G", model.feature_layers), ("C", model.fc), ("D", discriminator)):
            for layer in module.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    self.handles.append(layer.register_forward_hook(self.hook(tag)))

    def hook(self, tag):
        def add(module, inputs, output):
            per_output = module.in_features if isinstance(module, torch.nn.Linear) else (
                module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1])
            self.counts[tag] += output.numel() * per_output
        return add

    def remove(self):
        for handle in self.handles:
            handle.remove()


def train(seed, fold, restart_incomplete=False):
    check_environment()
    require(torch.cuda.is_available(), "Fixed implementation requires CUDA; no silent device/batch downgrade")
    root = run_root(seed, fold)
    if (root / "TRAINING_COMPLETE.json").exists():
        complete_training(root, current_implementation=True)
        print(f"Already complete and validated: {root}")
        return
    attempts = sorted(root.glob("attempt_*")) if root.exists() else []
    require(not attempts or restart_incomplete, "Interrupted attempt retained. Use --restart-incomplete to restart from the fixed seed")
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"attempt_{len(attempts) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    start = time.perf_counter()
    epoch = step = 0
    try:
        # Diagnostics run in separate processes; no probe iterator in this training stream.
        set_seed(seed)
        source, target, source_ids = training_loaders(fold)
        config = make_config(seed, fold, source_ids)
        write_json(attempt / "config.json", config)
        model = ASDAResNet18(pretrained=True).cuda()
        discriminator = Discriminator().cuda()
        optimizer, scheduler = optimizer_scheduler(model, discriminator)
        initial_bn = bn_counts(model)
        macs = ForwardMACs(model, discriminator)
        parameters = {"G": sum(p.numel() for p in model.feature_layers.parameters()),
                      "C": sum(p.numel() for p in model.fc.parameters()),
                      "D": sum(p.numel() for p in discriminator.parameters())}
        target_iter = iter(target)  # Deliberately OUTSIDE the source epoch loop.
        prepare_seconds = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        training_start = time.perf_counter()
        save_seconds = data_seconds = update_seconds = 0.0
        target_cycles = 0
        for epoch in range(1, 41):
            data_start = time.perf_counter()
            for xs, ys, source_batch_ids in source:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_cycles += 1
                    target_iter = iter(target)
                    target_batch = next(target_iter)
                loaded_seconds = time.perf_counter() - data_start
                data_seconds += loaded_seconds
                torch.cuda.synchronize()
                update_start = time.perf_counter()
                stats = train_step(model, discriminator, optimizer, scheduler, xs, ys, target_batch, "cuda")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - update_start
                update_seconds += elapsed
                step += 1
                if step == 1:
                    macs.remove()
                stats.update(epoch=epoch, global_step=step, target_iterator_cycle=target_cycles,
                             source_ids=list(source_batch_ids), target_ids=target_batch["ids"].tolist(),
                             ra_operations=target_batch["operations"], data_augmentation_seconds=loaded_seconds,
                             update_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved())
                append_json(attempt / "steps.jsonl", stats)
                data_start = time.perf_counter()
            require(step == epoch * 6, "Epoch step count differs")
            if epoch in EPOCHS:
                save_start = time.perf_counter()
                torch.save(checkpoint_payload(model, discriminator, config, epoch, step), attempt / f"asda_epoch{epoch:03d}.pth")
                save_seconds += time.perf_counter() - save_start
            print(f"seed={seed} fold={fold} epoch={epoch}/40 step={step}/240 loss={stats['loss']:.6f}", flush=True)
        require(step == 240, "Incomplete training updates")
        require(all(bn_counts(model)[key] == value + 240 for key, value in initial_bn.items()), "Wrong final BN update count")
        cost = {"actual_updates": step, "actual_source_images": step * 64,
                "actual_target_original_images": step * 64, "actual_target_augmented_images": step * 192,
                "actual_G_C_forward_images": step * 320, "actual_D_examples": step * 320,
                "source_batches_per_epoch": 6, "target_full_batches_per_iterator": 8,
                "target_iterator_renewals": target_cycles, "parameters": parameters,
                "first_step_forward_MACs_conv_linear": macs.counts,
                "full_training_forward_MACs_conv_linear": {k: v * step for k, v in macs.counts.items()},
                "backward_MACs_estimate_conv_linear": {k: 2 * v * step for k, v in macs.counts.items()},
                "MAC_scope": "Observed Conv/Linear shapes; 1 MAC = multiply+accumulate; excludes BN, activations, pooling, outer product, loss, optimizer; backward 2x is analytic estimate, not measured",
                "prepare_seconds": prepare_seconds, "training_wall_seconds_including_save_and_logging": time.perf_counter() - training_start,
                "data_augmentation_seconds": data_seconds, "synchronized_update_seconds": update_seconds,
                "checkpoint_save_seconds": save_seconds, "total_seconds_before_final_validation": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "inference": "G/C only; 553 images/checkpoint; last5 uses2765 G/C image forwards; no D/RA/AdaBN",
                "old_comparison": "Counts are actual ASDA, not old fixed-batch budgets or the old1.72 ratio"}
        write_training_complete(root, attempt, config, cost)
        print(f"Validated training completion: {attempt}", flush=True)
    except BaseException as exc:
        write_json(attempt / "INTERRUPTED.json", {"time": datetime.datetime.now().astimezone().isoformat(),
                   "epoch": epoch, "completed_steps": step, "exception": repr(exc), "traceback": traceback.format_exc(),
                   "restart_policy": "Preserve this attempt; restart whole run from fixed seed; never select by score"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.fold, args.restart_incomplete)


if __name__ == "__main__":
    main()
