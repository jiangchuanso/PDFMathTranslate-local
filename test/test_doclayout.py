import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pdf2zh.doclayout import (
    OnnxModel,
    YoloResult,
    YoloBox,
    _cpu_fingerprint,
    _is_optimized_model_usable,
)


class TestOnnxModel(unittest.TestCase):
    @patch("onnx.load")
    @patch("onnxruntime.InferenceSession")
    def setUp(self, mock_inference_session, mock_onnx_load):
        # Mock ONNX model metadata
        mock_model = MagicMock()
        mock_model.metadata_props = [
            MagicMock(key="stride", value="32"),
            MagicMock(key="names", value="['class1', 'class2']"),
        ]
        mock_onnx_load.return_value = mock_model

        # Initialize OnnxModel with a fake path
        self.model_path = "fake_model_path.onnx"
        self.model = OnnxModel(self.model_path)

    def test_stride_property(self):
        # Test that stride is correctly set from model metadata
        self.assertEqual(self.model.stride, 32)

    def test_resize_and_pad_image(self):
        # Create a dummy image (100x200)
        image = np.ones((100, 200, 3), dtype=np.uint8)
        resized_image = self.model.resize_and_pad_image(image, 1024)

        # Validate the output shape
        self.assertEqual(resized_image.shape[0], 512)
        self.assertEqual(resized_image.shape[1], 1024)

        # Check that padding has been added
        padded_height = resized_image.shape[0] - image.shape[0]
        padded_width = resized_image.shape[1] - image.shape[1]
        self.assertGreater(padded_height, 0)
        self.assertGreater(padded_width, 0)

    def test_scale_boxes(self):
        img1_shape = (1024, 1024)  # Model input shape
        img0_shape = (500, 300)  # Original image shape
        boxes = np.array([[512, 512, 768, 768]])  # Example bounding box

        scaled_boxes = self.model.scale_boxes(img1_shape, boxes, img0_shape)

        # Verify the output is scaled correctly
        self.assertEqual(scaled_boxes.shape, boxes.shape)
        self.assertTrue(np.all(scaled_boxes <= max(img0_shape)))

    def test_predict(self):
        # Mock model inference output
        mock_output = np.random.random((1, 300, 6))
        self.model.model.run.return_value = [mock_output]

        # Create a dummy image
        image = np.ones((500, 300, 3), dtype=np.uint8)

        results = self.model.predict(image)

        # Validate predictions
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], YoloResult)
        self.assertGreater(len(results[0].boxes), 0)
        self.assertIsInstance(results[0].boxes[0], YoloBox)


class TestYoloResult(unittest.TestCase):
    def test_yolo_result(self):
        # Example prediction data
        boxes = [
            [100, 200, 300, 400, 0.9, 0],
            [50, 100, 150, 200, 0.8, 1],
        ]
        names = ["class1", "class2"]

        result = YoloResult(boxes, names)

        # Validate the number of boxes and their order by confidence
        self.assertEqual(len(result.boxes), 2)
        self.assertGreater(result.boxes[0].conf, result.boxes[1].conf)
        self.assertEqual(result.names, names)


class TestYoloBox(unittest.TestCase):
    def test_yolo_box(self):
        # Example box data
        box_data = [100, 200, 300, 400, 0.9, 0]

        box = YoloBox(box_data)

        # Validate box properties
        self.assertEqual(box.xyxy, box_data[:4])
        self.assertEqual(box.conf, box_data[4])
        self.assertEqual(box.cls, box_data[5])


class TestOptimizedModelCache(unittest.TestCase):
    """The optimized graph embeds CPU-specific kernels, so it may only be
    reused on the machine that produced it (see _cpu_fingerprint)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.optimized_path = os.path.join(self.tmpdir.name, "model.onnx.optimized")
        self.meta_path = self.optimized_path + ".cpu"

    def _write(self, path, content):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def test_fingerprint_is_stable(self):
        self.assertEqual(_cpu_fingerprint(), _cpu_fingerprint())
        self.assertNotEqual(_cpu_fingerprint(), "")

    def test_accepts_graph_from_same_cpu(self):
        self._write(self.optimized_path, "graph")
        self._write(self.meta_path, _cpu_fingerprint())

        self.assertTrue(_is_optimized_model_usable(self.optimized_path, self.meta_path))

    def test_rejects_graph_from_other_cpu(self):
        self._write(self.optimized_path, "graph")
        self._write(self.meta_path, "fingerprint-of-another-machine")

        self.assertFalse(
            _is_optimized_model_usable(self.optimized_path, self.meta_path)
        )

    def test_rejects_graph_without_provenance(self):
        self._write(self.optimized_path, "graph")

        self.assertFalse(
            _is_optimized_model_usable(self.optimized_path, self.meta_path)
        )

    def test_rejects_missing_graph(self):
        self.assertFalse(
            _is_optimized_model_usable(self.optimized_path, self.meta_path)
        )

    @patch("onnx.load")
    @patch("onnxruntime.InferenceSession")
    def test_foreign_graph_is_dropped_and_regenerated(
        self, mock_inference_session, mock_onnx_load
    ):
        mock_model = MagicMock()
        mock_model.metadata_props = [
            MagicMock(key="stride", value="32"),
            MagicMock(key="names", value="['class1']"),
        ]
        mock_onnx_load.return_value = mock_model
        self._write(self.optimized_path, "foreign graph")
        self._write(self.meta_path, "fingerprint-of-another-machine")

        OnnxModel(self.optimized_path[: -len(".optimized")])

        # The foreign graph must not be handed to ONNX Runtime, it is deleted
        # and the original model is optimized again for the local CPU.
        self.assertFalse(os.path.exists(self.optimized_path))
        self.assertFalse(os.path.exists(self.meta_path))
        session_model_path = mock_inference_session.call_args[0][0]
        self.assertEqual(session_model_path, self.optimized_path[: -len(".optimized")])


if __name__ == "__main__":
    unittest.main()
