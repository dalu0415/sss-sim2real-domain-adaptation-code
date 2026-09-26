"""CPU tests for the ASDA comparator (src/extra/asda) and its archived-run evidence (evidence/asda).

No image data, network access or GPU is needed: the mechanism tests use synthetic tensors and a small stand-in
network for the fixed 64/64/3 training step, and the evidence tests recompute the reported summaries from the
tracked per-run records.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ASDA_ROOT = SRC_ROOT / "extra" / "asda"
EVIDENCE = REPO_ROOT / "evidence" / "asda"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(ASDA_ROOT))

import artifacts  # noqa: E402
import common  # noqa: E402
import data  # noqa: E402
import evaluate_asda  # noqa: E402
import model  # noqa: E402
import scoring  # noqa: E402
import train_asda  # noqa: E402
from evaluation.core import classification_metrics  # noqa: E402
from evaluation.statistics import paired_seed_differences, paired_t_difference  # noqa: E402


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(name: str):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _per_run_records() -> list[dict]:
    return [_load(f"per_run/asda_s{seed}_f{fold}.json") for seed in range(5) for fold in range(5)]


_SAVED_THREADS = []


def setUpModule() -> None:
    # The tensors here are small; one intra-op thread avoids thread oversubscription on a busy machine.
    _SAVED_THREADS.append(torch.get_num_threads())
    torch.set_num_threads(1)


def tearDownModule() -> None:
    torch.set_num_threads(_SAVED_THREADS.pop())


@functools.lru_cache(maxsize=1)
def _cdan_trainer():
    """The public CDAN trainer, loaded from its file (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("_cdan_trainer", SRC_ROOT / "run2_s5_cdan" / "train_cdan_sgd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StandIn(nn.Module):
    """Small network with the attributes train_step and optimizer_scheduler use (feature_layers, fc, BN)."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=16, stride=16), nn.BatchNorm2d(4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 512),
        )
        self.fc = nn.Linear(512, 2)
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape))
        features = self.feature_layers(x).flatten(1)
        return features, self.fc(features)


class ASDAMechanismTests(unittest.TestCase):
    def test_reliable_cluster_and_signed_entropy_use_the_last_view(self) -> None:
        original = torch.tensor([[4.0, 0.0], [4.0, 0.0]])  # both base predictions are class A
        views = torch.tensor([
            [[3.0, 0.0], [2.0, 0.0], [0.0, 1.0]],   # A, A, B: two agreements -> reliable
            [[3.0, 0.0], [0.0, 2.0], [0.0, 1.5]],   # A, B, B: one agreement -> unreliable
        ])
        signed, reliable, last_agrees = model.signed_entropy(original, views)
        self.assertEqual(reliable.tolist(), [True, False])
        self.assertEqual(last_agrees.tolist(), [False, False])

        def entropy(logits):
            p = logits.softmax(-1)
            return float(-(p * p.log()).sum())

        expected = (entropy(views[0, -1]) - entropy(views[1, -1])) / 2
        self.assertAlmostEqual(signed.item(), expected, places=6)

        # The entropy gradient reaches only the last view; the gate itself passes no gradient.
        views_grad, original_grad = views.clone().requires_grad_(True), original.clone().requires_grad_(True)
        model.signed_entropy(original_grad, views_grad)[0].backward()
        last = views.clone()[:, -1].requires_grad_(True)
        p = last.softmax(-1)
        (-(p * p.log()).sum(-1) * torch.tensor([1.0, -1.0])).mean().backward()
        self.assertFalse(views_grad.grad[:, :2].any())
        torch.testing.assert_close(views_grad.grad[:, -1], last.grad)
        self.assertTrue(original_grad.grad is None or not original_grad.grad.any())
        three = torch.tensor([[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
        self.assertTrue(model.signed_entropy(original[:1], three)[1].item())
        with self.assertRaises(ValueError):
            model.signed_entropy(original, views[:, :2])

    def test_conditional_input_detaches_classifier_probabilities(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(4, 512, requires_grad=True)
        logits = torch.randn(4, 2, requires_grad=True)
        joint = model.conditional_input(features, logits)
        self.assertEqual(tuple(joint.shape), (4, 1024))
        probs = logits.softmax(-1).detach()
        torch.testing.assert_close(joint, torch.cat([probs[:, :1] * features, probs[:, 1:] * features], dim=1))
        joint.sum().backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(features.grad)

    def test_domain_terms_average_each_domain_separately(self) -> None:
        logits = torch.tensor([2.0, -1.0, 0.5, 0.0, 3.0])
        source, target = model.domain_terms(logits, 2)
        expected_source = sum(math.log1p(math.exp(-x)) for x in (2.0, -1.0)) / 2
        expected_target = sum(math.log1p(math.exp(x)) for x in (0.5, 0.0, 3.0)) / 3
        self.assertAlmostEqual(source.item(), expected_source, places=6)
        self.assertAlmostEqual(target.item(), expected_target, places=6)
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                model.domain_terms(logits, bad)

    def test_gradient_reversal_negates_the_incoming_gradient(self) -> None:
        x = torch.tensor([0.5, -2.0, 3.0], requires_grad=True)
        y = model.Reverse.apply(x)
        torch.testing.assert_close(y.detach(), x.detach())
        (y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([-1.0, -2.0, -3.0]))

    def test_source_weights_optimizer_and_schedule(self) -> None:
        effective = [(1 - 0.999) / (1 - 0.999 ** n) for n in (72, 320)]
        expected = [w / sum(effective) * 2 for w in effective]
        weights = model.source_weights("cpu")
        torch.testing.assert_close(weights, torch.tensor(expected, dtype=torch.float32))
        self.assertGreater(weights[0].item(), weights[1].item())

        torch.manual_seed(0)
        optimizer, scheduler = model.optimizer_scheduler(_StandIn(), model.Discriminator())
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.001, 0.01, 0.01])
        for group in optimizer.param_groups:
            self.assertEqual((group["momentum"], group["nesterov"], group["weight_decay"]), (0.9, True, 5e-4))
        seen = {}
        for step in range(1, 301):
            optimizer.step()  # no gradients: parameters are unchanged, but the scheduler order is realistic
            scheduler.step()
            if step in (120, 240, 300):
                seen[step] = [g["lr"] for g in optimizer.param_groups]
        for step, factor in ((120, 6 ** -0.75), (240, 11 ** -0.75), (300, 11 ** -0.75)):
            for got, base in zip(seen[step], (0.001, 0.01, 0.01)):
                self.assertAlmostEqual(got, base * factor, places=12)

    def test_randaugment_convention_and_view_quantisation(self) -> None:
        augment = data.TracedRandAugment()
        self.assertEqual((augment.num_ops, augment.magnitude, augment.num_magnitude_bins), (3, 2, 31))
        self.assertEqual(augment.interpolation, data.T.InterpolationMode.NEAREST)
        space = augment._augmentation_space(31, (224, 224))
        self.assertEqual(len(space), 14)
        torch.manual_seed(3)
        image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        out, trace = augment(image)
        self.assertEqual((out.dtype, tuple(out.shape), len(trace)), (torch.uint8, (3, 224, 224), 3))
        for item in trace:
            magnitudes, _ = space[item["op"]]
            expected = float(magnitudes[2].item()) if magnitudes.ndim else 0.0
            self.assertAlmostEqual(abs(item["magnitude"]), abs(expected), places=6)
        for name, (magnitudes, _) in space.items():  # every operation of the pool runs on a uint8 image
            magnitude = float(magnitudes[2].item()) if magnitudes.ndim else 0.0
            result = data._apply_op(image, name, magnitude, interpolation=augment.interpolation, fill=[0.0] * 3)
            self.assertEqual((result.dtype, tuple(result.shape)), (torch.uint8, (3, 224, 224)), name)
        with self.assertRaises(ValueError):
            augment(image.float())
        for seed in range(20):  # the traced copy draws and applies exactly what torchvision's RandAugment does
            torch.manual_seed(seed)
            traced, _ = augment(image)
            torch.manual_seed(seed)
            self.assertTrue(torch.equal(traced, data.T.RandAugment.forward(augment, image)), seed)

        class Unchanged:
            def __call__(self, view):
                return view.clone(), []

        base = torch.zeros(3, 224, 224)
        base[0, 0, :3] = torch.tensor([2.0, -1.0, 100.4 / 255])
        untouched = base.clone()
        original, views, traces = data.target_cluster(base, Unchanged())
        self.assertTrue(torch.equal(base, untouched))
        self.assertEqual((tuple(original.shape), tuple(views.shape), len(traces)), ((3, 224, 224), (3, 3, 224, 224), 3))
        mean = torch.tensor(data.MEAN)[:, None, None]
        std = torch.tensor(data.STD)[:, None, None]
        restored = (views[0] * std + mean) * 255
        torch.testing.assert_close(restored[0, 0, :3], torch.tensor([255.0, 0.0, 100.0]), atol=1e-3, rtol=0)
        torch.testing.assert_close(original, data.NORMALIZE(base))

    def test_common_settings_match_the_cdan_trainer(self) -> None:
        cdan = _cdan_trainer()
        image =Image.fromarray(np.random.default_rng(0).integers(0, 256, (240, 260), dtype=np.uint8)).convert("RGB")
        for ours, theirs in ((data.train_transform(), cdan.build_train_transform()),
                             (data.eval_transform(), cdan.build_eval_transform())):
            torch.manual_seed(4)
            expected = theirs(image)
            torch.manual_seed(4)
            self.assertTrue(torch.equal(ours(image), expected))
        self.assertTrue(torch.equal(model.source_weights("cpu"), cdan.cb_wce_weights([72, 320], device="cpu")))
        optimizer, _ = model.optimizer_scheduler(_StandIn(), model.Discriminator())
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [cdan.BACKBONE_LR, cdan.FC_LR, cdan.DISC_LR])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], cdan.WEIGHT_DECAY)
        self.assertEqual((cdan.LR_DECAY_ALPHA, cdan.LR_DECAY_BETA, cdan.BATCH_SIZE), (10.0, 0.75, 64))

    def test_model_state_matches_the_cdan_architecture(self) -> None:
        torch.manual_seed(0)
        asda, cdan = model.ASDAResNet18(pretrained=False), _cdan_trainer().CDANResNet18(pretrained=False)
        self.assertEqual(set(asda.state_dict()), set(cdan.state_dict()))
        cdan.load_state_dict(asda.state_dict(), strict=True)
        asda.load_state_dict(_cdan_trainer().CDANResNet18(pretrained=False).state_dict(), strict=True)

    def test_joint_batch_keeps_each_view_next_to_its_base_image(self) -> None:
        xs = -torch.arange(1, 65, dtype=torch.float32).view(64, 1, 1, 1).expand(64, 3, 224, 224)
        original = (1000 + torch.arange(64, dtype=torch.float32)).view(64, 1, 1, 1).expand(64, 3, 224, 224)
        views = (2000 + 10 * torch.arange(64).view(64, 1) + torch.arange(3).view(1, 3)).float()
        views = views.view(64, 3, 1, 1, 1).expand(64, 3, 3, 224, 224)
        joint = model.joint_batch(xs, {"original": original, "views": views})
        markers = joint[:, 0, 0, 0]
        torch.testing.assert_close(markers[:64], xs[:, 0, 0, 0])
        torch.testing.assert_close(markers[64:128], original[:, 0, 0, 0])
        # train_step reshapes rows 128.. to (64, 3, ...): row 128 + 3i + v must be view v of base image i
        torch.testing.assert_close(markers[128:].view(64, 3), views[:, :, 0, 0, 0])

    def test_training_step_on_cpu_with_a_small_stand_in_network(self) -> None:
        def run_once():
            torch.manual_seed(11)
            network, discriminator = _StandIn(), model.Discriminator()
            optimizer, scheduler = model.optimizer_scheduler(network, discriminator)
            generator = torch.Generator().manual_seed(5)
            xs = torch.rand(64, 3, 224, 224, generator=generator)
            ys = torch.cat([torch.zeros(18), torch.ones(46)]).long()
            target = {"original": torch.rand(64, 3, 224, 224, generator=generator),
                      "views": torch.rand(64, 3, 3, 224, 224, generator=generator)}
            before, classifier = model.bn_counts(network), network.fc.weight.detach().clone()
            stats = model.train_step(network, discriminator, optimizer, scheduler, xs, ys, target, "cpu")
            after = model.bn_counts(network)
            self.assertEqual(network.calls, [(320, 3, 224, 224)])  # one joint forward of the whole batch
            self.assertFalse(torch.equal(classifier, network.fc.weight))
            return stats, before, after

        first, before, after = run_once()
        second, _, _ = run_once()
        self.assertEqual((first["n_source"], first["n_target_original"], first["n_target_views"], first["joint_images"]),
                         (64, 64, 192, 320))
        self.assertTrue(all(after[k] == before[k] + 1 for k in before))
        parts = first["source_ce"] + first["domain_source_mean"] + first["domain_target_mean"] + first["signed_entropy"]
        self.assertAlmostEqual(first["loss"], parts, places=5)
        self.assertTrue(math.isfinite(first["loss"]))
        self.assertEqual(first["lr_used"], [0.001, 0.01, 0.01])
        for got, base in zip(first["lr_next"], (0.001, 0.01, 0.01)):
            self.assertAlmostEqual(got, base * (1 + 10 / 240) ** -0.75, places=12)
        self.assertEqual(first["loss"], second["loss"])
        self.assertEqual(first["gradient_norms_G_C_D"], second["gradient_norms_G_C_D"])
        with self.assertRaises(ValueError):
            model.joint_batch(torch.rand(8, 3, 224, 224), {"original": torch.rand(8, 3, 224, 224),
                                                            "views": torch.rand(8, 3, 3, 224, 224)})

    def test_resnet18_contract_without_downloading_weights(self) -> None:
        torch.manual_seed(0)
        network = model.ASDAResNet18(pretrained=False)
        self.assertEqual(len(model.bn_counts(network)), 20)
        network.eval()
        with torch.no_grad():
            features, logits = network(torch.rand(2, 3, 224, 224))
        self.assertEqual((tuple(features.shape), tuple(logits.shape)), ((2, 512), (2, 2)))
        discriminator = model.Discriminator()
        self.assertEqual(sum(p.numel() for p in discriminator.parameters()), 2100225)

    def test_checkpoint_payload_and_fixed_epoch_validation(self) -> None:
        torch.manual_seed(0)
        network, discriminator = model.ASDAResNet18(pretrained=False), model.Discriminator()
        for module in network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.num_batches_tracked.fill_(36 * 6)
        config = {"seed": 1, "fold": 2, "inputs": {"klsg_unlabeled_manifest.txt": "0" * 64}}
        buffer = io.BytesIO()
        torch.save(artifacts.checkpoint_payload(network, discriminator, config, 36, 216), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        artifacts.validate_checkpoint(state, config, 36)
        for saved, live in ((state["model_state_dict"], network.state_dict()),
                            (state["ad_net_state_dict"], discriminator.state_dict())):
            self.assertEqual(set(saved), set(live))
            self.assertTrue(all(torch.equal(saved[k], live[k]) for k in live))
        rejected = [(state, config, 35), (state, config, 37), (state, dict(config, changed=True), 36),
                    (dict(state, model_state_dict={k: v + 1 if k.endswith("num_batches_tracked") else v
                                                   for k, v in state["model_state_dict"].items()}), config, 36)]
        tampered = [("global_step", 217), ("seed", 0), ("fold", 0), ("config_sha256", "0" * 64),
                    ("classes", ["ship", "airplane"]), ("target_ids", list(range(552))), ("manifest_sha256", "1" * 64),
                    ("schema", "other")]
        rejected += [(dict(state, **{key: value}), config, 36) for key, value in tampered]
        rejected.append(({k: v for k, v in state.items() if k != "ad_net_state_dict"}, config, 36))
        for bad_state, bad_config, epoch in rejected:
            with self.assertRaises(ValueError):
                artifacts.validate_checkpoint(bad_state, bad_config, epoch)

    def test_five_checkpoint_average_and_prediction_validation(self) -> None:
        rng = np.random.default_rng(0)
        predictions = [{"epoch": epoch, "ids": list(range(553)), "classes": ["airplane", "ship"],
                        "probs": rng.dirichlet([1.0, 1.0], size=553).astype(np.float32)}
                       for epoch in range(36, 41)]
        mean = artifacts.average_five_predictions(predictions)
        self.assertEqual(mean.dtype, np.float64)
        np.testing.assert_array_equal(mean, np.stack([p["probs"].astype(np.float64) for p in predictions]).mean(0))
        with self.assertRaises(ValueError):
            artifacts.average_five_predictions(predictions[::-1])
        with self.assertRaises(ValueError):
            artifacts.average_five_predictions(predictions[:4])
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(list(range(552)), ["airplane", "ship"], predictions[0]["probs"][:552])
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(list(range(553)), ["ship", "airplane"], predictions[0]["probs"])
        swapped = [dict(p) for p in predictions]
        swapped[2]["ids"] = [1, 0] + list(range(2, 553))  # same IDs, different order
        with self.assertRaises(ValueError):
            artifacts.average_five_predictions(swapped)
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(list(range(553)), ["airplane", "ship"], predictions[0]["probs"] * 2)

    def test_scoring_agrees_with_the_paper_evaluation_module(self) -> None:
        rng = np.random.default_rng(7)
        tied = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.5])  # airplane probabilities with ties at 0.5
        cases = [(np.array([0, 0, 1, 1, 1, 0]), np.stack([tied, 1 - tied], axis=1))]
        cases += [(rng.integers(0, 2, size=553), rng.dirichlet([1.0, 3.0], size=553)) for _ in range(20)]
        for y_true, probs in cases:
            ours = scoring.adjudication_metrics(y_true, probs.argmax(1), probs)
            reference = classification_metrics(y_true, probs)
            self.assertAlmostEqual(ours["macro_f1"], reference["macro_f1"], places=12)
            self.assertAlmostEqual(ours["airplane_pr_auc"], reference["airplane_pr_auc"], places=12)
            self.assertEqual(ours["confusion_matrix"], np.asarray(reference["confusion_matrix"]).tolist())
            for name in ("airplane", "ship"):
                for key in ("precision", "recall", "f1"):
                    self.assertAlmostEqual(ours["per_class"][name][key], reference["per_class"][name][key], places=12)
        values_a = {s: float(v) for s, v in enumerate(rng.random(5))}
        values_b = {s: float(v) for s, v in enumerate(rng.random(5))}
        arm = lambda values: {"per_seed_points": [{"seed": s, "values": {"m": v}} for s, v in values.items()]}
        ours = scoring.paired_delta(arm(values_a), arm(values_b), "m")
        reference = paired_t_difference(paired_seed_differences(values_a, values_b)["differences"])
        interval = reference["confidence_interval"]
        low, high = (interval["lower"], interval["upper"]) if isinstance(interval, dict) else interval
        self.assertAlmostEqual(ours["mean"], reference["mean_difference"], places=12)
        self.assertAlmostEqual(ours["sd"], reference["sample_sd"], places=12)
        self.assertAlmostEqual(ours["ci_lo"], low, places=12)
        self.assertAlmostEqual(ours["ci_hi"], high, places=12)
        self.assertAlmostEqual(scoring.T_CRIT, reference["t_critical"], places=12)


class ASDAEvidenceTests(unittest.TestCase):
    def test_evidence_checksums(self) -> None:
        lines = (EVIDENCE / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
        listed = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            listed[name] = digest
        present = {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob("*") if p.is_file()}
        self.assertEqual(set(listed), present - {"SHA256SUMS.txt"})
        for name, digest in listed.items():
            self.assertEqual(_sha256((EVIDENCE / name).read_bytes()), digest, name)

    def test_seed_summary_requires_all_25_unique_runs(self) -> None:
        records = _per_run_records()
        for bad in (records[:24], records[:24] + [records[0]], records + [records[0]]):
            with self.assertRaises(ValueError):
                evaluate_asda.seed_summary(bad)

    def test_forward_mac_counts_match_the_trainer(self) -> None:
        torch.manual_seed(0)
        network, discriminator = model.ASDAResNet18(pretrained=False).eval(), model.Discriminator().eval()
        counter = train_asda.ForwardMACs(network, discriminator)
        with torch.no_grad():
            network(torch.rand(2, 3, 224, 224))
            discriminator(torch.rand(2, 1024))
        counter.remove()
        per_image = {key: value // 2 for key, value in counter.counts.items()}
        self.assertEqual(per_image, {"G": 1_813_561_344, "C": 1_024, "D": 2_098_176})
        for record in _per_run_records():  # each update forwards 320 images through G, C and D
            self.assertEqual(record["training"]["forward_MACs_per_update_conv_linear"],
                             {key: 320 * value for key, value in per_image.items()})

    def test_summary_recomputes_from_the_per_run_records(self) -> None:
        records = _per_run_records()
        self.assertEqual([(r["seed"], r["fold"]) for r in records], [(s, f) for s in range(5) for f in range(5)])
        summary = _load("summary.json")
        candidate = evaluate_asda.seed_summary(records)
        self.assertEqual(candidate, summary["candidate"])
        reference = _load("reference_arms.json")
        self.assertEqual(_sha256((EVIDENCE / "reference_arms.json").read_bytes()), summary["reference_sha256"])
        self.assertEqual([arm["arm_id"] for arm in reference["arms"]], ["#1", "#2"])
        for arm in reference["arms"]:
            recomputed = [scoring.paired_delta(candidate, arm, m) for m in evaluate_asda.METRICS[:5]]
            self.assertEqual(recomputed, summary["paired_deltas_ASDA_minus_reference"][arm["arm_id"]])

    def test_values_reported_in_the_manuscript(self) -> None:
        summary = _load("summary.json")
        across = summary["candidate"]["across_seed"]
        pct = lambda x: round(100 * x, 2)
        table_6_e2 = {"macro_f1": (41.91, 20.15), "airplane_f1": (19.27, 2.82), "airplane_pr_auc": (32.16, 11.39),
                      "airplane_p": (53.89, 30.41), "airplane_r": (39.21, 41.67), "ship_f1": (64.55, 42.52),
                      "ship_p": (92.34, 4.86), "ship_r": (67.91, 45.46)}
        for metric, (mean, sd) in table_6_e2.items():
            self.assertEqual((pct(across[metric]["mean"]), pct(across[metric]["sd"])), (mean, sd), metric)
        per_seed = {p["seed"]: p["values"] for p in summary["candidate"]["per_seed_points"]}
        table_e3 = {0: (55.00, 16.04, 35.85), 1: (31.03, 22.63, 26.36), 2: (56.71, 19.29, 42.33),
                    3: (55.42, 16.93, 41.09), 4: (11.37, 21.44, 15.18)}
        for seed, values in table_e3.items():
            got = tuple(pct(per_seed[seed][m]) for m in ("macro_f1", "airplane_f1", "airplane_pr_auc"))
            self.assertEqual(got, values, seed)
        contrasts = summary["paired_deltas_ASDA_minus_reference"]
        expected = {("#1", "macro_f1"): (-20.79, -46.24, 4.67), ("#1", "airplane_f1"): (-19.33, -23.73, -14.92),
                    ("#1", "airplane_pr_auc"): (-5.40, -19.09, 8.29),
                    ("#2", "macro_f1"): (-26.67, -51.05, -2.30), ("#2", "airplane_f1"): (-25.39, -30.26, -20.53),
                    ("#2", "airplane_pr_auc"): (-12.13, -24.70, 0.45)}  # Table E.3 prints #2 as (AdaBN) - ASDA
        for (arm_id, metric), values in expected.items():
            record = [c for c in contrasts[arm_id] if c["metric"] == metric][0]
            self.assertEqual((pct(record["mean"]), pct(record["ci_lo"]), pct(record["ci_hi"])), values)
        table_6_references = {"#1": {"macro_f1": (62.69, 1.87), "airplane_f1": (38.59, 1.95), "airplane_pr_auc": (37.57, 1.99)},
                              "#2": {"macro_f1": (68.58, 1.07), "airplane_f1": (44.66, 1.64), "airplane_pr_auc": (44.29, 2.24)}}
        for arm in _load("reference_arms.json")["arms"]:
            for metric, (mean, sd) in table_6_references[arm["arm_id"]].items():
                values = [p["values"][metric] for p in arm["per_seed_points"]]
                self.assertEqual((pct(float(np.mean(values))), pct(float(np.std(values, ddof=1)))), (mean, sd))

    def test_prediction_bias_cost_and_order_of_events(self) -> None:
        records = _per_run_records()
        dominant = [r["dominant_class"] for r in records]
        self.assertEqual((dominant.count("ship"), dominant.count("airplane")), (17, 8))
        self.assertGreaterEqual(min(r["dominant_share"] for r in records), 0.95)
        loops = np.array([r["training"]["training_loop_seconds"] for r in records])
        q1, median, q3 = np.percentile(loops, [25, 50, 75])
        self.assertEqual([round(v, 2) for v in (median, q1, q3, loops.min(), loops.max())],
                         [468.60, 389.14, 470.27, 258.61, 474.03])
        prediction = np.array([r["prediction"]["target_last5_seconds"] for r in records])
        q1, median, q3 = np.percentile(prediction, [25, 50, 75])
        self.assertEqual([round(v, 2) for v in (median, q1, q3, prediction.min(), prediction.max())],
                         [6.92, 6.81, 6.99, 6.55, 7.34])
        peaks = {(r["training"]["peak_allocated_bytes"], r["training"]["peak_reserved_bytes"]) for r in records}
        self.assertEqual(len(peaks), 1)  # Appendix F: the same peaks in all 25 training runs
        allocated, reserved = peaks.pop()
        self.assertEqual((round(allocated / 2 ** 30, 3), round(reserved / 2 ** 30, 3)), (7.034, 11.26))
        counts = {tuple(sorted(r["training"]["forward_MACs_per_update_conv_linear"].items())) for r in records}
        self.assertEqual(len(counts), 1)
        macs = dict(counts.pop())
        # Section 2.4 convention: Conv/Linear MACs plus the conditional outer product (320 images x 2 classes x 512
        # features), 2 FLOPs per MAC, backward counted as twice forward, 240 updates; Table 5a in 10^12 FLOPs.
        training = 2 * 3 * (sum(macs.values()) + 320 * 2 * 512) * 240
        prediction = 2 * (macs["G"] + macs["C"]) // 320 * 553 * 5  # G/C only, 553 images, five checkpoints
        self.assertEqual((round(training / 1e12, 3), round(prediction / 1e12, 3)), (836.657, 10.029))
        execution = _load("provenance.json")["execution"]
        training_end = execution["training_stage_utc"][1]
        for record in records:
            self.assertEqual(record["training"]["updates"], 240)
            self.assertEqual(sorted(record["checkpoint_sha256"]), [f"asda_epoch{e:03d}.pth" for e in range(36, 41)])
            self.assertGreater(record["label_access"]["first_read_utc"], training_end)

    def test_protocol_and_inputs_match_the_archived_runs(self) -> None:
        provenance = _load("provenance.json")
        self.assertEqual(common.PROTOCOL, provenance["protocol"])
        self.assertEqual(common.canonical_hash(common.PROTOCOL), provenance["protocol_sha256"])
        self.assertTrue(all(r["protocol_sha256"] == provenance["protocol_sha256"] for r in _per_run_records()))
        self.assertEqual(common.INIT_SHA256, provenance["initialization"]["sha256"])
        for name, recorded in provenance["inputs"]["recorded_sha256"].items():
            lf = (REPO_ROOT / "data" / "split" / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertIn(recorded, {_sha256(lf), _sha256(lf.replace(b"\n", b"\r\n"))}, name)
            self.assertEqual(_sha256(lf), provenance["inputs"]["public_sha256"][name], name)

    def test_records_carry_no_local_paths(self) -> None:
        local = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|[\\/](Users|home)[\\/]")
        for path in EVIDENCE.rglob("*"):
            if path.is_file():
                match = local.search(path.read_text(encoding="ascii"))
                self.assertIsNone(match, f"{path.name}: {match and match.group(0)!r}")


if __name__ == "__main__":
    unittest.main()
