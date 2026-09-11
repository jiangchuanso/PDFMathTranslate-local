import abc
import contextlib
import hashlib
import logging
import os
import platform
import sys

import cv2
import numpy as np
import ast
from babeldoc.assets.assets import get_doclayout_onnx_model_path

try:
    import onnx
    import onnxruntime
except ImportError as e:
    if "DLL load failed" in str(e):
        raise OSError(
            "Microsoft Visual C++ Redistributable is not installed. "
            "Download it at https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from e
    raise

logger = logging.getLogger(__name__)

_BACKEND_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_preferred_backend: str | None = None


def set_backend(name: str) -> None:
    """Set the ONNX Runtime execution provider backend.

    Args:
        name: One of 'auto', 'cpu', 'cuda', 'dml'.
    """
    global _preferred_backend
    _preferred_backend = None if name == "auto" else name


class DocLayoutModel(abc.ABC):
    @staticmethod
    def load_onnx():
        model = OnnxModel.from_pretrained()
        return model

    @staticmethod
    def load_available():
        return DocLayoutModel.load_onnx()

    @property
    @abc.abstractmethod
    def stride(self) -> int:
        """Stride of the model input."""
        pass

    @abc.abstractmethod
    def predict(self, image, imgsz=1024, **kwargs) -> list:
        """
        Predict the layout of a document page.

        Args:
            image: The image of the document page.
            imgsz: Resize the image to this size. Must be a multiple of the stride.
            **kwargs: Additional arguments.
        """
        pass


class YoloResult:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, boxes, names):
        self.boxes = [YoloBox(data=d) for d in boxes]
        self.boxes.sort(key=lambda x: x.conf, reverse=True)
        self.names = names


class YoloBox:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, data):
        self.xyxy = data[:4]
        self.conf = data[-2]
        self.cls = data[-1]


def _cpu_fingerprint() -> str:
    """Identify the CPU feature set that an optimized ONNX graph depends on.

    ONNX Runtime bakes the kernels and tensor layouts it picked for the local
    CPU (for example ``com.microsoft.nchwc`` conv ops, AVX512 layouts) into the
    optimized graph. Such a graph is only safe to reuse on the CPU that
    produced it, otherwise inference can crash with SIGFPE/SIGSEGV, which is
    not catchable as a Python exception.
    """
    identity = [sys.platform, platform.machine(), platform.processor() or ""]
    if os.name == "nt":
        identity.append(os.environ.get("PROCESSOR_IDENTIFIER", ""))
    elif sys.platform == "linux":
        cpuinfo_keys = {"model name", "vendor_id", "flags", "features"}
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    key, sep, value = line.partition(":")
                    key = key.strip().lower()
                    if not sep or key not in cpuinfo_keys:
                        continue
                    if key in ("flags", "features"):
                        value = " ".join(sorted(set(value.split())))
                    identity.append(f"{key}={value.strip()}")
        except OSError:
            pass
    return hashlib.sha256("\n".join(identity).encode("utf-8")).hexdigest()[:16]


def _is_optimized_model_usable(optimized_path: str, meta_path: str) -> bool:
    """Whether a pre-optimized graph may be reused on this machine."""
    if not os.path.exists(optimized_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as fh:
            recorded = fh.read().strip()
    except OSError:
        # Graph without provenance (produced by an older version) is untrusted.
        logger.warning(
            "Optimized model has no CPU record, re-optimizing: %s", optimized_path
        )
        return False
    if recorded == _cpu_fingerprint():
        return True
    logger.warning(
        "Ignoring optimized model built on a different CPU: %s", optimized_path
    )
    return False


class OnnxModel(DocLayoutModel):
    def __init__(self, model_path: str):
        model_path = str(model_path)
        self.model_path = model_path

        # Extract metadata without full model deserialization
        model = onnx.load(model_path, load_external_data=False)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])
        del model  # free memory before creating session

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if _preferred_backend and _preferred_backend in _BACKEND_PROVIDERS:
            providers = _BACKEND_PROVIDERS[_preferred_backend]
        else:
            providers = onnxruntime.get_available_providers()

        # Providers like CoreML generate compiled nodes that cannot be
        # serialized, so only cache the optimized graph for CPU-only.
        compiled_providers = {"CoreMLExecutionProvider", "TensorrtExecutionProvider"}
        can_cache = not compiled_providers.intersection(providers)
        optimized_path = None
        optimized_meta_path = None
        if can_cache:
            optimized_path = model_path + ".optimized"
            optimized_meta_path = optimized_path + ".cpu"
            if _is_optimized_model_usable(optimized_path, optimized_meta_path):
                model_path = optimized_path
            else:
                # Reuse only graphs this machine produced, see _cpu_fingerprint.
                for stale in (optimized_path, optimized_meta_path):
                    with contextlib.suppress(OSError):
                        os.remove(stale)
                sess_options.optimized_model_filepath = optimized_path

        try:
            self.model = onnxruntime.InferenceSession(
                model_path, sess_options, providers=providers
            )
        except Exception:
            if optimized_path is None or model_path != optimized_path:
                raise
            # A cached optimized graph may be unusable on this machine, e.g. it
            # was produced on x86 and relies on com.microsoft.nchwc ops. Drop it
            # and fall back to the original model.
            logger.warning("Discarding unusable optimized model: %s", model_path)
            with contextlib.suppress(OSError):
                os.remove(model_path)
            fallback_options = onnxruntime.SessionOptions()
            fallback_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.model = onnxruntime.InferenceSession(
                model_path[: -len(".optimized")], fallback_options, providers=providers
            )
        if (
            optimized_meta_path is not None
            and os.path.exists(optimized_path)
            and not os.path.exists(optimized_meta_path)
        ):
            # ONNX Runtime just wrote a graph for this very CPU, record it so
            # other machines sharing the cache directory do not reuse it.
            with contextlib.suppress(OSError):
                with open(optimized_meta_path, "w", encoding="utf-8") as fh:
                    fh.write(_cpu_fingerprint())
        logger.info("ONNX Runtime providers: %s", self.model.get_providers())

    @staticmethod
    def from_pretrained():
        pth = get_doclayout_onnx_model_path()
        return OnnxModel(pth)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        """
        Resize and pad the image to the specified size, ensuring dimensions are multiples of stride.

        Parameters:
        - image: Input image
        - new_shape: Target size (integer or (height, width) tuple)
        - stride: Padding alignment stride, default 32

        Returns:
        - Processed image
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape

        # Calculate scaling ratio
        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))

        # Resize image
        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding size and align to stride multiple
        pad_w = (new_w - resized_w) % self.stride
        pad_h = (new_h - resized_h) % self.stride
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        # Add padding
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for,
                in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """

        # Calculate scaling ratio
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        # Calculate padding size
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        # Remove padding and scale boxes
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=1024, **kwargs):
        # Preprocess input image
        orig_h, orig_w = image.shape[:2]
        pix = self.resize_and_pad_image(image, new_shape=imgsz)
        pix = np.transpose(pix, (2, 0, 1))  # CHW
        pix = np.expand_dims(pix, axis=0)  # BCHW
        pix = pix.astype(np.float32) / 255.0  # Normalize to [0, 1]
        new_h, new_w = pix.shape[2:]

        # Run inference
        preds = self.model.run(None, {"images": pix})[0]

        # Postprocess predictions
        preds = preds[preds[..., 4] > 0.25]
        preds[..., :4] = self.scale_boxes(
            (new_h, new_w), preds[..., :4], (orig_h, orig_w)
        )
        return [YoloResult(boxes=preds, names=self._names)]


class ModelInstance:
    value: OnnxModel = None
