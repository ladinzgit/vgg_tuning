# train_vgg.py
import argparse, time, os, math, random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# 0. Global: Fixed Seed
FIXED_SEED = 1202

def set_seed(seed: int = FIXED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1. CLI args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_root", type=str, default="../../data/")
    p.add_argument("--logdir", type=str, default="./logs")
    p.add_argument("--suite", type=str, default="struct6",
                   choices=["struct6", "crop5", "cutmix5"])
    p.add_argument("--max_exps", type=int, default=None)
    return p.parse_args()


# 2. Model (Configurable VGG16-ish)
class VGG(nn.Module):
    def __init__(self, widths, pool_pattern, num_classes=100, bn=True):
        super().__init__()
        blocks = [2,2,3,3,3]  
        feats = []
        in_ch = 3  # RGB input
        for b, num_convs in enumerate(blocks):
            out_ch = widths[b]
            for _ in range(num_convs):
                conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not bn)
                if bn:
                    feats += [conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                else:
                    feats += [conv, nn.ReLU(inplace=True)]
                in_ch = out_ch
            s = pool_pattern[b]
            feats += [nn.MaxPool2d(kernel_size=2, stride=s)]
        self.features = nn.Sequential(*feats)
        self.avgpool  = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(widths[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 3. Data & Transforms
def compute_mean_std(root):
    tmp = torchvision.datasets.CIFAR100(root=root, train=True, transform=T.ToTensor(), download=True)
    loader = DataLoader(tmp, batch_size=5000, shuffle=False, num_workers=2)
    mean = 0.0; std = 0.0; total = 0
    for imgs, _ in loader:
        b = imgs.size(0)
        imgs = imgs.view(b, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std  += imgs.std(2).sum(0)
        total += b
    mean /= total; std /= total
    return mean, std

def build_transforms(mean, std, crop_size):
    aug = [T.RandomHorizontalFlip()]
    if crop_size < 32:
        aug += [T.RandomCrop(crop_size), T.Resize(32)]
    else:
        aug += [T.RandomCrop(32)]
    train_tf = T.Compose(aug + [T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return train_tf, test_tf


# 4. CutMix utils
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2); y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2); y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2

def apply_cutmix(images, targets, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    b, _, H, W = images.size()
    index = torch.randperm(b).to(images.device)
    x1,y1,x2,y2 = rand_bbox(W,H,lam)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    targets_a, targets_b = targets, targets[index]
    lam = 1 - ((x2-x1)*(y2-y1)/(W*H))  # 실제 비율
    return images, targets_a, targets_b, lam


# 5. Train / Eval
def train_one_epoch(model, loader, criterion, optimizer, device, use_cutmix=False, alpha=1.0):
    model.train()
    t0 = time.perf_counter()
    running_loss, running_correct, total = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_cutmix:
            imgs, la, lb, lam = apply_cutmix(imgs, labels, alpha)
            out = model(imgs)
            loss = lam * criterion(out, la) + (1 - lam) * criterion(out, lb)
        else:
            out = model(imgs)
            loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            pred = out.argmax(1)
            if use_cutmix:
                running_correct += (pred==la).sum().item()
            else:
                running_correct += (pred==labels).sum().item()
        total += imgs.size(0)
    t1 = time.perf_counter()
    return running_loss/total, running_correct/total, (t1-t0)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
        loss += criterion(out, labels).item() * imgs.size(0)
        correct += (out.argmax(1)==labels).sum().item()
        total += imgs.size(0)
    return loss/total, correct/total


# 6. Experiment Suites (>=5 configs)
def get_suite(suite_name: str):
    """
    각 아이템은 (tag, cfg_dict)
    cfg_dict keys: widths, pool, crop, cutmix, alpha
    """
    if suite_name == "struct6":
        widths_list = [
            "48,96,192,384,384",
            "64,128,256,512,512",
            "96,192,384,768,768",
        ]
        pools = ["2,2,2,2,2", "2,2,1,1,1"]
        exps = []
        for w in widths_list:
            for p in pools:
                tag = f"W[{w}]_P[{p}]"
                exps.append((tag, dict(
                    widths=w, pool=p, crop=32, cutmix=False, alpha=1.0
                )))
        return exps  

    if suite_name == "crop5":
        # RandomCrop size 5개(20/22/24/28/32) → 5개 비교
        sizes = [20, 22, 24, 28, 32]
        exps = []
        for sz in sizes:
            tag = f"CROP{sz}"
            exps.append((tag, dict(
                widths="64,128,256,512,512",
                pool="2,2,2,2,2",
                crop=sz,
                cutmix=False,
                alpha=1.0
            )))
        return exps  # 5

    if suite_name == "cutmix5":
        # (cutmix on/off) × (crop 28/32) + 추가로 crop24(without cutmix) = 5
        base = [
            ("CM0_C32", dict(widths="64,128,256,512,512", pool="2,2,2,2,2", crop=32, cutmix=False, alpha=1.0)),
            ("CM1_C32", dict(widths="64,128,256,512,512", pool="2,2,2,2,2", crop=32, cutmix=True , alpha=1.0)),
            ("CM0_C28", dict(widths="64,128,256,512,512", pool="2,2,2,2,2", crop=28, cutmix=False, alpha=1.0)),
            ("CM1_C28", dict(widths="64,128,256,512,512", pool="2,2,2,2,2", crop=28, cutmix=True , alpha=1.0)),
            ("CM0_C24", dict(widths="64,128,256,512,512", pool="2,2,2,2,2", crop=24, cutmix=False, alpha=1.0)),
        ]
        return base  # 5

    raise ValueError(f"Unknown suite: {suite_name}")


# 7. Main (multi-run orchestrator)
def main():
    args = get_args()
    set_seed(FIXED_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 미리 평균/표준편차 1회 계산
    mean, std = compute_mean_std(args.data_root)

    # 실행할 설정 목록
    suite_exps = get_suite(args.suite)
    if args.max_exps is not None:
        suite_exps = suite_exps[:args.max_exps]
    # logs/<suite>/...
    suite_dir = os.path.join(args.logdir, args.suite)
    os.makedirs(suite_dir, exist_ok=True)

    # 전체 요약 CSV
    summary_csv = os.path.join(suite_dir, "summary.csv")
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w") as f:
            f.write("tag,params,epochs,best_val_acc,test_acc,avg_epoch_sec,widths,pool,crop,cutmix\n")

    # 공통: test set은 변하지 않음 (항상 동일한 normalize 사용)
    _, test_tf = build_transforms(mean, std, crop_size=32)
    test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, transform=test_tf, download=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    for tag, cfg in suite_exps:
        # 각 실험마다 seed 재설정으로 공정한 비교
        set_seed(FIXED_SEED)
        
        # 각 실험별 디렉토리
        run_name = f"{tag}"
        outdir = os.path.join(suite_dir, run_name)
        os.makedirs(outdir, exist_ok=True)

        # Transforms / Dataset per experiment (crop마다 학습 분포가 달라짐)
        train_tf, _ = build_transforms(mean, std, crop_size=cfg["crop"])
        full_train = torchvision.datasets.CIFAR100(root=args.data_root, train=True, transform=train_tf, download=True)
        n_train = int(0.9 * len(full_train))
        n_val   = len(full_train) - n_train
        train_set, val_set = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(FIXED_SEED))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=256,         shuffle=False, num_workers=4, pin_memory=True)

        # Model
        widths = list(map(int, cfg["widths"].split(",")))
        pool   = list(map(int, cfg["pool"].split(",")))
        model = VGG(widths=widths, pool_pattern=pool, num_classes=100, bn=True).to(device)
        nparams = count_params(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/3)

        # Logging
        writer = SummaryWriter(outdir)
        csv_path = os.path.join(outdir, "log.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_sec,params\n")

        best_val, best_state = 0.0, None
        epoch_times = []

        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc, t_sec = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_cutmix=cfg["cutmix"], alpha=cfg["alpha"]
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Acc/train", tr_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("Time/epoch_sec", t_sec, epoch)

            with open(csv_path, "a") as f:
                f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.4f},{val_loss:.6f},{val_acc:.4f},{int(t_sec)},{nparams}\n")

            epoch_times.append(t_sec)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                torch.save({"state_dict": best_state, "val_acc": best_val, "cfg": cfg},
                           os.path.join(outdir, "best.pth"))

        # Test with best
        if best_state is not None:
            model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        with open(os.path.join(outdir, "test.txt"), "w") as f:
            f.write(f"best_val_acc={best_val:.4f}\n")
            f.write(f"test_loss={test_loss:.6f}\n")
            f.write(f"test_acc={test_acc:.4f}\n")

        # Append to suite summary
        avg_epoch_sec = sum(epoch_times)/len(epoch_times) if epoch_times else 0.0
        with open(summary_csv, "a") as f:
            f.write(f"{tag},{nparams},{args.epochs},{best_val:.4f},{test_acc:.4f},{avg_epoch_sec:.2f},"
                    f"{cfg['widths']},{cfg['pool']},{cfg['crop']},{int(cfg['cutmix'])}\n")

        writer.close()

        # 메모리 정리
        del model, train_loader, val_loader, train_set, val_set, full_train
        torch.cuda.empty_cache()

    print(f"[DONE] Suite '{args.suite}' finished. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
