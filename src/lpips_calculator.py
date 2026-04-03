"""
LPIPS Calculator
Computes LPIPS (Learned Perceptual Image Patch Similarity) between
a camera-captured holographic reconstruction and a target image.

LPIPS range: 0.0 (identical) to ~1.0 (very different)
"""

import numpy as np
from pathlib import Path
import torch
import lpips
import cv2
from PIL import Image


class LPIPSCalculator:
    """
    Wraps the lpips library for perceptual similarity evaluation.
    """

    def __init__(
        self,
        network: str = "alex",
        spatial: bool = False,
        resize_to: int = 256,
    ):
        """
        Args:
            network:   Backbone network: 'alex' (faster) or 'vgg' (more accurate)
            spatial:   If True, return per-pixel LPIPS map; if False, return scalar
            resize_to: Resize both images to this size before computation
                       (0 = no resize). Recommended: 64–512.
        """
        self.network = network
        self.spatial = spatial
        self.resize_to = resize_to

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = lpips.LPIPS(net=network, spatial=spatial).to(self.device)
        self._model.eval()

    def compute(
        self,
        captured: "str | np.ndarray",
        target: "str | np.ndarray",
    ) -> float:
        """
        Compute LPIPS between a captured reconstruction and a target image.

        Args:
            captured: Captured image - file path or numpy array (H,W,3) BGR uint8
            target:   Target image   - file path or numpy array (H,W,3) BGR uint8

        Returns:
            LPIPS score (float). Lower = more similar.
        """
        img_cap = self._load_and_preprocess(captured)
        img_tgt = self._load_and_preprocess(target)

        with torch.no_grad():
            score = self._model(img_cap, img_tgt)

        if self.spatial:
            return float(score.mean().item())
        else:
            return float(score.item())

    def compute_spatial(
        self,
        captured: "str | np.ndarray",
        target: "str | np.ndarray",
    ) -> np.ndarray:
        """
        Compute spatial LPIPS map (per-pixel similarity).

        Returns:
            numpy array (H, W) with per-pixel LPIPS scores.
        """
        model_spatial = lpips.LPIPS(net=self.network, spatial=True).to(self.device)
        model_spatial.eval()

        img_cap = self._load_and_preprocess(captured)
        img_tgt = self._load_and_preprocess(target)

        with torch.no_grad():
            score_map = model_spatial(img_cap, img_tgt)

        return score_map.squeeze().cpu().numpy()

    def _load_and_preprocess(self, image: "str | np.ndarray") -> torch.Tensor:
        """Load image and convert to LPIPS-compatible tensor [-1, 1]."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale -> RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # Assume BGR (OpenCV convention)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image[:, :, :3]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize if requested
        if self.resize_to > 0:
            img_rgb = cv2.resize(
                img_rgb, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA
            )

        # Convert to float tensor [0, 1] then normalize to [-1, 1]
        tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
        # Shape: (H, W, 3) -> (1, 3, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    @staticmethod
    def compute_psnr(
        captured: "str | np.ndarray",
        target: "str | np.ndarray",
        resize_to: int = 0,
    ) -> float:
        """Compute PSNR as a supplementary metric."""
        def load(img):
            if isinstance(img, str):
                arr = cv2.imread(img)
            else:
                arr = img.copy()
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if resize_to > 0:
                arr = cv2.resize(arr, (resize_to, resize_to))
            return arr.astype(np.float64)

        cap = load(captured)
        tgt = load(target)
        mse = np.mean((cap - tgt) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def compute_ssim(
        captured: "str | np.ndarray",
        target: "str | np.ndarray",
        resize_to: int = 0,
    ) -> float:
        """Compute SSIM as a supplementary metric (requires scikit-image)."""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            return float("nan")

        def load(img):
            if isinstance(img, str):
                arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            else:
                if img.ndim == 3:
                    arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    arr = img
            if resize_to > 0:
                arr = cv2.resize(arr, (resize_to, resize_to))
            return arr

        cap = load(captured)
        tgt = load(target)
        score, _ = ssim(cap, tgt, full=True)
        return float(score)
