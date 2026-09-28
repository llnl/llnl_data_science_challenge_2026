"""Chat-parser regression tests."""

import unittest
from time import perf_counter

from part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi import _chat_response, store


class ChatListingTests(unittest.TestCase):
    def test_dashboard_summary_is_complete_and_geometry_free(self):
        payload = store.dashboard_records

        self.assertEqual(len(payload), len(store.defect_by_strut))
        self.assertIn("missing_score", payload[0])
        self.assertIn("needs_review", payload[0])
        self.assertNotIn("start_x_vox", payload[0])

    def test_indexed_compact_records_are_fast(self):
        keys = store.defect_by_strut["_strut_key"].iloc[:1000]
        started = perf_counter()
        for key in keys:
            store._compact_strut_record(key)

        self.assertLess(perf_counter() - started, 2.0)

    def test_strut_records_include_full_defect_summary(self):
        record = store._compact_strut_record(str(store.centerlines.iloc[0]["_strut_key"]))

        self.assertIn("defect_summary", record)
        self.assertIn("missing_score", record["defect_summary"])
        self.assertIn("needs_review", record["defect_summary"])

    def test_active_strut_supports_station_deviation_question(self):
        strut_id = int(store.defect_by_strut.iloc[0]["strut_id"])
        response = _chat_response("Where is the largest deviation?", active_strut_id=strut_id)

        self.assertEqual(response["strut_id"], strut_id)
        self.assertIn("position_fraction", response["reply"])

    def test_bent_listing_returns_primary_defect_ids_with_a_cap(self):
        response = _chat_response("Which struts are classified as bent?")
        expected = store.defect_by_strut[
            store.defect_by_strut["primary_defect"].astype(str).str.casefold() == "bent"
        ]["strut_id"].astype(int).tolist()

        self.assertEqual(response["field"], "primary_defect")
        self.assertEqual(response["value"], "Bent")
        self.assertEqual(response["total"], len(expected))
        self.assertEqual(response["strut_ids"], expected[:100])
        self.assertLessEqual(len(response["strut_ids"]), 100)
        self.assertEqual(response["references"], ["/struts?primary_defect=Bent"])

    def test_explicit_classification_uses_stage2_field(self):
        response = _chat_response("List struts with Stage 2 classification Missing_Intentional")

        self.assertEqual(response["field"], "stage2_classification")
        self.assertEqual(response["value"], "Missing_Intentional")
        self.assertEqual(response["total"], 87)
        self.assertEqual(response["references"], ["/struts?classification=Missing_Intentional"])

    def test_bare_nominal_prefers_classification(self):
        response = _chat_response("Show Nominal struts")

        self.assertEqual(response["field"], "stage2_classification")
        self.assertEqual(response["value"], "Nominal")

    def test_missing_subtypes_accept_human_readable_spacing(self):
        for prompt in (
            "Which struts are Missing Intentional?",
            "Which struts are missing-intentional?",
            "Which struts are MISSING_INTENTIONAL?",
        ):
            response = _chat_response(prompt)
            self.assertEqual(response["field"], "stage2_classification")
            self.assertEqual(response["value"], "Missing_Intentional")
            self.assertEqual(response["total"], 87)

    def test_bare_missing_requests_a_subtype(self):
        response = _chat_response("Which struts are missing?")

        self.assertNotIn("strut_ids", response)
        self.assertEqual(
            response["missing_subtypes"],
            {"Missing_Intentional": 87, "Missing_Unintentional": 331},
        )
        self.assertIn("Please specify", response["reply"])


if __name__ == "__main__":
    unittest.main()
