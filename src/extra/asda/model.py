"""ASDA-SEL-01/A: single shared-BN forward, conditional DA, signed last-view entropy."""
from common import init_state_dict, require
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ASDAResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Instantiate the SAME base architecture (including discarded 1000-way FC),
        # then load the digest-verified torchvision V1 weights; no old adapted checkpoint.
        base = resnet18(weights=None)
        if pretrained:
            base.load_state_dict(init_state_dict(), strict=True)
        self.in_features = 512
        self.feature_layers = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        self.fc = nn.Linear(512, 2)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                require(module.eps == 1e-5 and module.momentum == 0.1 and module.affine
                        and module.track_running_stats, "Unexpected BN contract")

    def forward(self, x):
        features = self.feature_layers(x).flatten(1)
        return features, self.fc(features)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def conditional_input(features, logits):
    probs = logits.softmax(-1).detach()
    return torch.bmm(probs.unsqueeze(2), features.unsqueeze(1)).flatten(1)


def domain_terms(domain_logits, n_source):
    require(domain_logits.ndim == 1 and 0 < n_source < len(domain_logits), "Invalid domain boundary")
    source = F.binary_cross_entropy_with_logits(domain_logits[:n_source], torch.ones_like(domain_logits[:n_source]))
    target = F.binary_cross_entropy_with_logits(domain_logits[n_source:], torch.zeros_like(domain_logits[n_source:]))
    return source, target


def signed_entropy(original_logits, view_logits):
    require(view_logits.shape == (len(original_logits), 3, 2), "Invalid sample/view/class axes")
    with torch.no_grad():
        original_prediction = original_logits.argmax(-1)
        matches = view_logits.argmax(-1).eq(original_prediction[:, None])
        reliable = 2 * matches.sum(1) > 3
    logp = F.log_softmax(view_logits[:, -1, :], dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    signed = torch.where(reliable, entropy, -entropy).mean()
    return signed, reliable, matches[:, -1]


def source_weights(device):
    w = [(1 - 0.999) / (1 - 0.999 ** n) for n in (72, 320)]
    return torch.tensor([x / sum(w) * 2 for x in w], dtype=torch.float32, device=device)


def optimizer_scheduler(model, discriminator):
    groups = [{"params": list(model.feature_layers.parameters()), "lr": 0.001, "name": "G"},
              {"params": list(model.fc.parameters()), "lr": 0.01, "name": "C"},
              {"params": list(discriminator.parameters()), "lr": 0.01, "name": "D"}]
    ids = [id(p) for group in groups for p in group["params"]]
    require(len(ids) == len(set(ids)), "Duplicated optimizer parameter")
    optimizer = torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (1 + 10 * min(s, 240) / 240) ** -0.75)
    return optimizer, scheduler


def joint_batch(xs, target):
    original, views = target["original"], target["views"]
    require(xs.shape == (64, 3, 224, 224) and original.shape == xs.shape
            and views.shape == (64, 3, 3, 224, 224), "Fixed 64/64/3 layout required")
    return torch.cat([xs, original, views.flatten(0, 1)], dim=0)


def bn_counts(model):
    return {name: int(module.num_batches_tracked.item()) for name, module in model.named_modules()
            if isinstance(module, nn.BatchNorm2d)}


def train_step(model, discriminator, optimizer, scheduler, xs, ys, target, device):
    model.train()
    discriminator.train()
    x = joint_batch(xs, target).to(device)
    ys = ys.to(device)
    require(ys.shape == (64,) and torch.all((ys == 0) | (ys == 1)).item(), "Invalid source labels")
    before = bn_counts(model)
    lr_used = [group["lr"] for group in optimizer.param_groups]
    optimizer.zero_grad(set_to_none=True)
    features, logits = model(x)  # One and only one G/C forward, N=320.
    ce = F.cross_entropy(logits[:64], ys, weight=source_weights(device))
    entropy, reliable, last_agrees = signed_entropy(logits[64:128], logits[128:].reshape(64, 3, 2))
    domain_logits = discriminator(Reverse.apply(conditional_input(features, logits)))
    domain_source, domain_target = domain_terms(domain_logits, 64)
    loss = ce + domain_source + domain_target + entropy
    require(torch.isfinite(loss).item(), "Nonfinite loss")
    loss.backward()
    norms = []
    for group in optimizer.param_groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        require(len(grads) == len(group["params"]) and all(torch.isfinite(g).all().item() for g in grads),
                f"Missing/nonfinite gradient in {group['name']}")
        norms.append(float(torch.sqrt(sum(g.detach().square().sum() for g in grads)).item()))
    after = bn_counts(model)
    require(all(after[k] == before[k] + 1 for k in before), "BN did not update exactly once")
    optimizer.step()
    scheduler.step()
    require(all(torch.isfinite(p).all().item() for module in (model, discriminator) for p in module.parameters()),
            "Nonfinite parameter after update")
    return {"loss": loss.item(), "source_ce": ce.item(), "domain_source_mean": domain_source.item(),
            "domain_target_mean": domain_target.item(), "signed_entropy": entropy.item(),
            "reliable_fraction": reliable.float().mean().item(),
            "last_agrees_fraction": last_agrees.float().mean().item(), "lr_used": lr_used,
            "lr_next": [group["lr"] for group in optimizer.param_groups], "grl": 1.0,
            "gradient_norms_G_C_D": norms, "finite": True, "bn_counts": after,
            "n_source": 64, "n_target_original": 64, "n_target_views": 192, "joint_images": 320}
