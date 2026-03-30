from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.serving import app


class ModelServingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")

    def test_predict_endpoint(self) -> None:
        payload = {
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127.0,
            "total_phenols": 2.8,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280_od315_of_diluted_wines": 3.92,
            "proline": 1065.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("predicted_class", body)
        self.assertIn("confidence", body)
        self.assertEqual(len(body["class_probabilities"]), 3)

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            "model_inference_requests_total" in response.text or "model_metadata" in response.text
        )


if __name__ == "__main__":
    unittest.main()
