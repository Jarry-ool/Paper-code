# zerone_gradcam.py
# =================
# 统一 Grad/Score/Eigen-CAM 脚本：
#   - 自动从 zerone_config 读取 IMG_OUT_ROOT / CLASSES
#   - 默认加载 IMG_OUT_ROOT 下的 resnet18_best_test.pt（若不存在则退回 resnet18_best.pt）
#   - 支持三种 CAM：Grad-CAM / Score-CAM / Eigen-CAM（--cam-type）
#   - 支持三种模式：
#       1) 单图：  python zerone_gradcam.py --image path/to/img.png
#       2) 文件夹：python zerone_gradcam.py --folder path/to/dir
#       3) 一键全跑：
#              python zerone_gradcam.py --split test
#              python zerone_gradcam.py --split val
#              python zerone_gradcam.py --split train
#              python zerone_gradcam.py --all
#      遍历 IMG_OUT_ROOT/<split>/<class>/*.png，统一在 IMG_OUT_ROOT/gradcam/... 下保存结果

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms

from zerone_config import IMG_OUT_ROOT, CLASSES

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------- 模型构建 & 权重加载 ----------------- #

def build_resnet18_for_gradcam(device: torch.device) -> torch.nn.Module:
    """
    构建与训练脚本一致的 ResNet18 结构，并自动加载权重：
    优先加载 IMG_OUT_ROOT/resnet18_best_test.pt，没有再用 resnet18_best.pt。
    """
    num_classes = len(CLASSES)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    root = Path(IMG_OUT_ROOT)
    ckpt_test = root / "resnet18_best_test.pt"
    ckpt_val  = root / "resnet18_best.pt"

    if ckpt_test.exists():
        print(f"[GradCAM] 加载测试集最优权重: {ckpt_test}")
        state = torch.load(ckpt_test, map_location=device)
        model.load_state_dict(state)
    elif ckpt_val.exists():
        print(f"[GradCAM] 加载验证集最优权重: {ckpt_val}")
        state = torch.load(ckpt_val, map_location=device)
        model.load_state_dict(state)
    else:
        print("[WARN] 未找到 resnet18_best_test.pt / resnet18_best.pt，将使用随机初始化模型（结果仅供调试）。")

    model.eval()
    return model


# ----------------- 三类 CAM 核心实现 ----------------- #

class ResNet18CAM:
    """
    针对 ResNet18 的 CAM 封装：
      - hook layer4 的 feature map 和 grad
      - Grad-CAM：通道权重=grad 的 GAP
      - Score-CAM：用上采样后的每个通道特征作为 mask 乘到输入，前向得到目标类 score 作为权重
      - Eigen-CAM：对激活 A (C×H×W) 做 PCA/SVD，取第一主成分投影为热图
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.target_activations = None
        self.target_grads = None

        # 注册 hook：取 layer4 的输出
        def forward_hook(module, inp, out):
            self.target_activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.target_grads = grad_out[0].detach()

        self.model.layer4.register_forward_hook(forward_hook)
        self.model.layer4.register_backward_hook(backward_hook)

        # 与训练脚本一致的几何预处理，避免额外归一化影响可视化直观性
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])

    # ---------- Integrated Gradients ----------
    def integrated_gradients(self, x: torch.Tensor, class_idx: int | None = None,
                             baseline: torch.Tensor | None = None,
                             steps: int = 50) -> np.ndarray:
        """
        计算 Integrated Gradients (IG) 显著图。
        对输入 x 与基准 baseline 之间做线性插值，累积梯度。
        返回 (H,W) 的 IG 热图。

        参考文献：Sundararajan et al., "Axiomatic Attribution for Deep Networks"【720121157950145†L48-L56】。
        """
        if baseline is None:
            baseline = torch.zeros_like(x, device=x.device)
        # 若未指定类别，则使用模型预测的最大类
        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
        # 生成插值输入
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
        # 累积梯度
        total_grad = torch.zeros_like(x, device=x.device)
        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)
            logits_interp = self.model(x_interp)
            score = logits_interp[0, class_idx]
            self.model.zero_grad()
            score.backward(retain_graph=True)
            if x_interp.grad is not None:
                total_grad += x_interp.grad.detach()
        avg_grad = total_grad / len(alphas)
        ig = (x - baseline) * avg_grad
        # 聚合各颜色通道，取绝对值平均作为灰度显著图
        ig_np = ig.squeeze(0).abs().mean(dim=0).detach().cpu().numpy()
        return ig_np

    # ---------- 共有工具 ----------

    def _preprocess(self, img_path: Path) -> Tuple[torch.Tensor, Image.Image]:
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)  # (1,3,150,150)
        return x.to(self.device), img

    @staticmethod
    def _to_cam_image(arr_2d: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        """把 2D CAM 数组归一化并插值到给定大小，返回 [0,1] numpy HxW"""
        cam = arr_2d.copy()
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        H, W = size_hw
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
        return np.asarray(cam_img, dtype=np.float32) / 255.0

    # ---------- Grad-CAM ----------

    def grad_cam(self, x: torch.Tensor, class_idx: int | None) -> np.ndarray:
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        A = self.target_activations  # (1,C,H,W)
        G = self.target_grads        # (1,C,H,W)
        weights = G.mean(dim=(2, 3), keepdim=True)           # (1,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=False)[0]     # (H,W)
        cam_np = cam.detach().cpu().numpy()
        return cam_np

    # ---------- Score-CAM ----------
    def score_cam(self, x: torch.Tensor, class_idx: int | None, topk: int | None = None) -> np.ndarray:
        """
        逐通道把激活上采样为 mask，乘到输入，前向得分作为权重。
        为控制推理量，可选 topk 通道（按激活能量排序）。
        """
        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            # 取一次前向的激活
            A = self.target_activations.detach()  # (1,C,H,W)
            C, H, W = A.shape[1:]

            # 选择 topk 通道（按通道能量排序）
            act_energy = A.view(1, C, -1).abs().mean(dim=2).squeeze(0)  # (C,)
            if topk is None or topk <= 0 or topk > C:
                idxs = torch.argsort(act_energy, descending=True)
            else:
                idxs = torch.argsort(act_energy, descending=True)[:topk]

            # 上采样到输入尺寸
            up = torch.nn.functional.interpolate(A[:, idxs, :, :], size=x.shape[-2:], mode="bilinear", align_corners=False)  # (1,K,H,W)
            up = up.squeeze(0)  # (K,H,W)
            K = up.shape[0]

            # 归一化到 [0,1] 作为 mask
            up_min = up.view(K, -1).min(dim=1)[0].view(K, 1, 1)
            up_max = up.view(K, -1).max(dim=1)[0].view(K, 1, 1)
            mask = (up - up_min) / (up_max - up_min + 1e-6)  # (K,H,W)

            # 对每个通道mask，乘到输入并前向，取目标类 softmax 作为权重
            scores = []
            for k in range(K):
                masked = x.clone()
                m = mask[k:k+1, ...]  # (1,H,W)
                masked = masked * m.unsqueeze(1)             # 按通道广播乘到 3 个颜色通道
                s = F.softmax(self.model(masked), dim=1)[0, class_idx].item()
                scores.append(s)
            w = torch.tensor(scores, device=x.device).view(K, 1, 1)  # (K,1,1)

            # 线性组合原激活（注意这里按 Score-CAM 论文，直接对“上采样后的通道图”加权求和亦可）
            cam = (w * up).sum(dim=0)  # (H,W)
            cam_np = cam.detach().cpu().numpy()
            return cam_np

    # ---------- Eigen-CAM ----------
    def eigen_cam(self) -> np.ndarray:
        """
        对激活 A (1,C,H,W) 做 SVD，取第一主成分对应的空间投影（等价于 A^T * v1）映射为 H×W。
        """
        A = self.target_activations.detach()[0]  # (C,H,W)
        C, H, W = A.shape
        M = A.view(C, -1).cpu().numpy()          # (C,HW)
        # SVD: M = U S V^T，取第一右奇异向量 v1（长度 HW）
        try:
            # 使用经济 SVD
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            v1 = Vt[0]                            # (HW,)
            cam = v1.reshape(H, W)
        except np.linalg.LinAlgError:
            cam = M.mean(axis=0).reshape(H, W)    # 退化：取均值
        return cam


# ----------------- 可视化 & 批量处理工具 ----------------- #

def overlay_cam_on_image(cam: np.ndarray, pil_img: Image.Image, alpha: float = 0.4) -> Image.Image:
    """将 [0,1] 的 CAM 叠加到原图上，返回彩色热力叠加图。"""
    # 使用 matplotlib colormap 生成热力
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]  # (H,W,3)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_uint8).convert("RGB")
    heatmap_img = heatmap_img.resize(pil_img.size, resample=Image.BILINEAR)
    blended = Image.blend(pil_img.convert("RGB"), heatmap_img, alpha=alpha)
    return blended

def _save_one(out_dir: Path, img_path: Path, suffix: str, blended: Image.Image):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_{suffix}.png"
    blended.save(out_path)
    print(f"[CAM] Saved -> {out_path}")

def run_one_image(cam_engine: ResNet18CAM, img_path: Path, out_root: Path, cam_type: str, freq_aggregate: bool = False):
    """
    对单张图片运行 CAM，并按 split/class 组织输出：
      - 输入图：IMG_OUT_ROOT/<split>/<class>/<name>.png
      - 输出图：IMG_OUT_ROOT/gradcam/<split>/<class>/<name>_{grad|score|eigen}.png
    """
    img_path = img_path.resolve()
    # 尝试从路径中解析 split / class
    cls_name = img_path.parent.name
    split_name = img_path.parent.parent.name  # train/val/test

    out_dir_base = out_root / split_name / cls_name

    x, pil_img = cam_engine._preprocess(img_path)
    with torch.no_grad():
        logits = cam_engine.model(x)
        class_idx = int(logits.argmax(dim=1).item())

    H, W = x.shape[-2], x.shape[-1]

    def _blend_and_save(cam_np: np.ndarray, tag: str):
        # 归一化并插值至输入尺寸
        cam01 = cam_engine._to_cam_image(cam_np, (H, W))
        blended = overlay_cam_on_image(cam01, pil_img, alpha=0.4)
        _save_one(out_dir_base, img_path, f"{tag}_cam", blended)
        # 可选：输出频率聚合曲线
        if freq_aggregate and cam_np.ndim == 2:
            try:
                import matplotlib.pyplot as plt
                freq_curve = cam_np.mean(axis=0)
                fig = plt.figure(figsize=(5.0, 3.2))
                ax = fig.add_subplot(111)
                ax.plot(freq_curve)
                ax.set_title(f"{tag.upper()} 频率重要度曲线")
                ax.set_xlabel("频率 Bin")
                ax.set_ylabel("重要度")
                ax.grid(True, alpha=0.3)
                freq_dir = out_dir_base / "freq"
                freq_dir.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(freq_dir / f"{img_path.stem}_{tag}_freq.png", dpi=200)
                plt.close(fig)
            except Exception:
                pass

    if cam_type in ("grad", "all"):
        cam_g = cam_engine.grad_cam(x, class_idx)
        _blend_and_save(cam_g, "grad")

    if cam_type in ("score", "all"):
        cam_s = cam_engine.score_cam(x, class_idx, topk=64)   # 可调 topk
        _blend_and_save(cam_s, "score")

    if cam_type in ("eigen", "all"):
        # 触发一次前向以拿到 A
        _ = cam_engine.model(x)
        cam_e = cam_engine.eigen_cam()
        _blend_and_save(cam_e, "eigen")

    if cam_type in ("ig", "all"):
        cam_ig = cam_engine.integrated_gradients(x, class_idx=class_idx, steps=50)
        _blend_and_save(cam_ig, "ig")


def collect_images_by_split(root: Path, split: str) -> List[Path]:
    """收集 IMG_OUT_ROOT/<split>/<class>/*.png"""
    out = []
    split_dir = root / split
    if not split_dir.exists():
        return out
    for cls in CLASSES:
        d = split_dir / cls
        if not d.exists():
            continue
        out.extend(sorted(d.glob("*.png")))
    return out


# ----------------- CLI ----------------- #

def main():
    parser = argparse.ArgumentParser(description="ZERONE ResNet18 CAM 可视化脚本")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--folder", type=str, default=None, help="包含若干 PNG 的文件夹")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="对某个 split (train/val/test) 下所有图片一键生成 CAM")
    parser.add_argument("--all", action="store_true",
                        help="对 train/val/test 三个 split 下的所有图片一键生成 CAM")
    parser.add_argument("--outdir", type=str, default=None,
                        help="输出根目录（默认：IMG_OUT_ROOT/gradcam）")
    parser.add_argument("--cam-type", type=str, default="grad",
                        choices=["grad", "score", "eigen", "ig", "all"],
                        help="选择 CAM 类型（支持 grad/score/eigen/ig/all，默认 grad）")
    parser.add_argument("--freq-aggregate", action="store_true",
                        help="将 2D 显著图沿纵轴平均，输出频率重要度曲线")

    args = parser.parse_args()

    root = Path(IMG_OUT_ROOT).resolve()
    # 默认输出到 IMG_OUT_ROOT/gradcam
    default_out = root / "gradcam"
    out_root = Path(args.outdir).resolve() if args.outdir else default_out
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_for_gradcam(device)
    cam_engine = ResNet18CAM(model, device)

    # --- 模式 1：单图 --- #
    if args.image is not None:
        img_path = Path(args.image)
        # 对单张图片执行 CAM，并根据参数输出频率曲线
        run_one_image(cam_engine, img_path, out_root, args.cam_type, freq_aggregate=args.freq_aggregate)
        return

    # --- 模式 2：指定文件夹 --- #
    if args.folder is not None:
        folder = Path(args.folder)
        imgs = sorted(list(folder.glob("*.png")))
        print(f"[CAM] 文件夹模式，共 {len(imgs)} 张图")
        for p in imgs:
            run_one_image(cam_engine, p, out_root, args.cam_type, freq_aggregate=args.freq_aggregate)
        return

    # --- 模式 3：一键 split / all --- #
    # 若未显式指定，则默认跑 test split
    splits: List[str] = []
    if args.all:
        splits = ["train", "val", "test"]
    elif args.split is not None:
        splits = [args.split]
    else:
        splits = ["test"]

    for sp in splits:
        imgs = collect_images_by_split(root, sp)
        print(f"[CAM] split={sp}，共 {len(imgs)} 张图")
        for p in imgs:
            run_one_image(cam_engine, p, out_root, args.cam_type, freq_aggregate=args.freq_aggregate)


if __name__ == "__main__":
    main()
