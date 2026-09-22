import unittest
from unittest.mock import patch

from PIL import Image

from utils import flag_logo_recognition


class FakeRecognizer:
    def __init__(self, similarity=0.8, class_name="United States"):
        self.similarity = similarity
        self.class_name = class_name

    def predict(self, image, mode):
        return {
            "id": "213",
            "class_key": "flags:213" if mode == "flag" else "Logo:213",
            "source_category": "flags" if mode == "flag" else "Logo",
            "class_name": self.class_name,
            "category": mode,
            "similarity": self.similarity,
            "score_type": "cosine_similarity",
        }


def make_target(object_name="flag"):
    return [{
        "bbox_id": 1,
        "bbox": [0, 0, 16, 16],
        "object_name": object_name,
        "probability": 0.9,
    }]


class FlagLogoRecognitionTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "white")

    @patch("utils.call_caption_api")
    def test_accepts_top1_when_similarity_reaches_threshold(self, caption_api):
        result = flag_logo_recognition(
            FakeRecognizer(similarity=0.8), "flag", self.image,
            make_target(), threshold=0.7, caption_api_url="http://caption",
        )
        caption_api.assert_not_called()
        self.assertEqual(result[0]["object_finegrained_name"], "United States")
        self.assertEqual(result[0]["flag"]["confidence"], 0.8)
        self.assertEqual(result[0]["flag"]["score_type"], "cosine_similarity")
        self.assertEqual(result[0]["flag"]["recognition_source"], "siglip2_two_mode")
        self.assertEqual(result[0]["object_finegrained_type"], "Flag")
        self.assertEqual(result[0]["object_finegrained_dataset"], "country-flags-in-the-wild")
        self.assertEqual(result[0]["object_finegrained_id"], "213")

    @patch("utils.call_caption_api", return_value="United States")
    def test_low_similarity_uses_caption_fallback(self, caption_api):
        result = flag_logo_recognition(
            FakeRecognizer(similarity=0.5, class_name="Liberia"), "flag",
            self.image, make_target(), threshold=0.7,
            caption_api_url="http://caption",
        )
        caption_api.assert_called_once()
        self.assertEqual(result[0]["object_finegrained_name"], "United States")
        self.assertEqual(result[0]["flag"]["recognition_source"], "caption_fallback")
        self.assertEqual(result[0]["flag"]["siglip2_top1_name"], "Liberia")
        self.assertEqual(result[0]["flag"]["confidence"], 0.5)
        self.assertNotIn("object_finegrained_type", result[0])
        self.assertNotIn("object_finegrained_id", result[0])

    @patch("utils.call_caption_api", side_effect=RuntimeError("offline"))
    def test_caption_failure_returns_siglip_top1(self, caption_api):
        result = flag_logo_recognition(
            FakeRecognizer(similarity=0.5, class_name="Liberia"), "flag",
            self.image, make_target(), threshold=0.7,
            caption_api_url="http://caption",
        )
        caption_api.assert_called_once()
        self.assertEqual(result[0]["object_finegrained_name"], "Liberia")
        self.assertEqual(result[0]["flag"]["recognition_source"], "siglip2_two_mode")
        self.assertNotIn("object_finegrained_type", result[0])

    @patch("utils.call_caption_api")
    def test_party_metadata_is_promoted_for_navigation(self, caption_api):
        recognizer = FakeRecognizer(similarity=0.8, class_name="Indian National Congress")
        recognizer.predict = lambda image, mode: {
            "id": "Q10225",
            "qid": "Q10225",
            "class_key": "Party:Q10225",
            "source_category": "Party",
            "class_name": "Indian National Congress",
            "category": mode,
            "similarity": 0.8,
            "score_type": "cosine_similarity",
        }
        result = flag_logo_recognition(
            recognizer, "logo", self.image, make_target("logo"),
            threshold=0.7, caption_api_url="http://caption",
        )
        caption_api.assert_not_called()
        self.assertEqual(result[0]["object_name"], "logo")
        self.assertEqual(result[0]["object_finegrained_type"], "Party")
        self.assertEqual(result[0]["object_finegrained_dataset"], "Party")
        self.assertEqual(result[0]["object_finegrained_id"], "Q10225")


if __name__ == "__main__":
    unittest.main()
