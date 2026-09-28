"""Interactive terminal client for the FastAPI dashboard chatbot."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from uuid import uuid4

try:
    import websocket
    WebSocketException = websocket.WebSocketException
except ImportError:  # Keep the help text available before dependencies are installed.
    websocket = None  # type: ignore[assignment]
    WebSocketException = Exception


DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8000/ws"


class ConnectionRecoveryError(RuntimeError):
    """Raised after a lost broker connection could not be recovered."""

GUIDANCE = """Supported prompts:

  Summary:
    How many struts are in the dataset?
    Summarize the defects.

  Review candidates:
    How many struts need review?
    List the struts needing review.

  Strut inspection:
    Inspect strut 42.
    Show details for strut 10.
    What stations does strut 7 have?

  Defect counts:
    How many struts have an Inflated defect?
    Count Nominal struts.

  Strut lists:
    Which struts are classified as Bent?
    List struts with Stage 2 classification Missing_Intentional.
    Which struts are Missing Intentional?
    Which struts are Missing Unintentional?

  Napari selection:
    Select strut 12.
    Highlight struts 12, 18, and 27.

Type 'help' to show this message again, or 'exit' to quit."""


def _websocket_url() -> str:
    return os.getenv("DASHBOARD_WEBSOCKET_URL", DEFAULT_WEBSOCKET_URL)


def connect_to_backend() -> websocket.WebSocket:
    """Connect to the broker and consume its initial state event."""
    if websocket is None:
        raise RuntimeError("Install the websocket-client package with 'pip install websocket-client'.")
    connection = websocket.create_connection(_websocket_url(), timeout=10)
    initial_event = json.loads(connection.recv())
    if initial_event.get("event_type") != "INIT_STATE":
        connection.close()
        raise RuntimeError("FastAPI connected but did not send an INIT_STATE event.")
    return connection


def _close_quietly(connection: websocket.WebSocket | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except (OSError, WebSocketException):
        pass


def request_chat(
    connection: websocket.WebSocket, message: str, request_id: str | None = None
) -> Dict[str, Any]:
    """Send one prompt and wait for its corresponding chat response."""
    connection.send(
        json.dumps(
            {
                "event_type": "CHAT_REQUEST",
                "data": {"message": message, "request_id": request_id or str(uuid4())},
            }
        )
    )

    while True:
        event = json.loads(connection.recv())
        event_type = event.get("event_type")
        data = event.get("data", {})

        # Selection is broadcast before the CHAT_RESPONSE. This lets the CLI
        # confirm that Napari was updated while still waiting for the reply.
        if event_type == "STRUTS_SELECTED":
            print(f"[Napari] Selected struts: {data.get('strut_ids', [])}")
        elif event_type == "CHAT_RESPONSE":
            return data
        elif event_type == "ERROR":
            raise RuntimeError(data.get("message", "FastAPI returned an error."))


def request_chat_with_recovery(
    connection: websocket.WebSocket, message: str
) -> tuple[Dict[str, Any], websocket.WebSocket]:
    """Retry one interrupted request after reconnecting to the broker.

    A Windows `WinError` 10053/10054 and a normal WebSocket close both surface
    through this transport-exception path.  Retrying only once avoids an
    unbounded loop or repeated side effects for selection prompts.
    """
    request_id = str(uuid4())
    try:
        return request_chat(connection, message, request_id), connection
    except (OSError, WebSocketException) as original_error:
        print(f"Broker connection was interrupted ({original_error}); reconnecting once...")
        _close_quietly(connection)
        try:
            recovered_connection = connect_to_backend()
            response = request_chat(recovered_connection, message, request_id)
            print("Broker connection recovered; retried the prompt.")
            return response, recovered_connection
        except (OSError, WebSocketException, json.JSONDecodeError, RuntimeError) as recovery_error:
            _close_quietly(locals().get("recovered_connection"))
            raise ConnectionRecoveryError(
                "Broker connection was lost and one reconnect attempt failed. "
                f"Original error: {original_error}. Retry error: {recovery_error}"
            ) from recovery_error


def print_response(response: Dict[str, Any]) -> None:
    reply = response.get("reply")
    if reply:
        print(f"Bot: {reply}")

    references = response.get("references", [])
    if references:
        print("References: " + ", ".join(str(reference) for reference in references))


def main() -> None:
    print("🤖 Strut Assistant Ready!\n" + "-" * 50)
    print(GUIDANCE)
    print()

    try:
        connection = connect_to_backend()
    except (OSError, WebSocketException, json.JSONDecodeError, RuntimeError) as exc:
        print(
            "Unable to connect to FastAPI. Start the broker with "
            "'uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app "
            f"--host localhost --port 8000'.\nDetails: {exc}"
        )
        return

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if user_input.casefold() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            if user_input.casefold() == "help":
                print(GUIDANCE)
                continue

            if connection is None:
                try:
                    connection = connect_to_backend()
                    print("Broker connection restored.")
                except (OSError, WebSocketException, json.JSONDecodeError, RuntimeError) as exc:
                    print(f"Unable to reconnect to FastAPI: {exc}\n")
                    continue

            try:
                response, connection = request_chat_with_recovery(connection, user_input)
                print_response(response)
                print()
            except ConnectionRecoveryError as exc:
                print(f"Unable to process the prompt: {exc}")
                print(
                    "Start the broker in PowerShell with "
                    "'uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app "
                    "--host 127.0.0.1 --port 8000', then try another prompt.\n"
                )
                connection = None
            except (json.JSONDecodeError, RuntimeError) as exc:
                print(f"Unable to process the prompt: {exc}\n")
    finally:
        _close_quietly(connection)


if __name__ == "__main__":
    main()
