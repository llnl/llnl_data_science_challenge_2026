"""Focused transport-recovery tests for the terminal chatbot."""

import unittest
from unittest.mock import patch

from part2.stage_4_interactive_dashboard_napari_chatbot import run_agent_cli


class FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ChatRecoveryTests(unittest.TestCase):
    def test_reconnects_once_and_reuses_the_same_request_id(self):
        original = FakeConnection()
        recovered = FakeConnection()
        calls = []

        def fake_request(connection, message, request_id):
            calls.append((connection, message, request_id))
            if connection is original:
                raise OSError(10053, "connection aborted")
            return {"reply": "The dataset contains 1 strut."}

        with patch.object(run_agent_cli, "request_chat", side_effect=fake_request), patch.object(
            run_agent_cli, "connect_to_backend", return_value=recovered
        ):
            response, connection = run_agent_cli.request_chat_with_recovery(original, "How many struts?")

        self.assertEqual(response["reply"], "The dataset contains 1 strut.")
        self.assertIs(connection, recovered)
        self.assertTrue(original.closed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:], calls[1][1:])

    def test_recovery_failure_raises_actionable_error(self):
        original = FakeConnection()
        with patch.object(run_agent_cli, "request_chat", side_effect=OSError(10053, "aborted")), patch.object(
            run_agent_cli, "connect_to_backend", side_effect=OSError(10061, "refused")
        ):
            with self.assertRaises(run_agent_cli.ConnectionRecoveryError) as context:
                run_agent_cli.request_chat_with_recovery(original, "summary")

        self.assertIn("10053", str(context.exception))
        self.assertIn("10061", str(context.exception))
        self.assertTrue(original.closed)


if __name__ == "__main__":
    unittest.main()
