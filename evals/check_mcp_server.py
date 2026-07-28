"""Confirm the MCP server starts and advertises its tools, without launching an agent.

Run this before a scored evaluation. Both ways the setup can break are silent:
a missing dependency stops the *whole* server rather than one tool, because
``skeleton_graph`` imports ``skan`` at module scope, and an interpreter or path
pointing at a different checkout starts a server that simply lacks the newer
tools. In both cases an agent sees no error -- it just quietly does the analysis
by hand, which measures nothing.

    python evals/check_mcp_server.py
    python evals/check_mcp_server.py --python /opt/anaconda3/envs/dssi_env/bin/python

Note this cannot be done by piping the three JSON-RPC messages in on stdin: the
server shuts down on EOF, and beats its own reply to ``tools/list``. The client
has to hold stdin open, which is why this is a script and not a one-liner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER = REPO_ROOT / "src" / "mcp_server.py"
EXPECTED_TOOLS = {
    "convert_tiff_volume",
    "graph_skeleton",
    "scan_lattice_junctions",
    "scan_lattice_struts",
    "segment_ct_dataset",
    "skeletonize",
    "visualize_junction_overlay",
    "visualize_slice",
}


def list_tools(interpreter: str, timeout: float) -> list[dict]:
    """Complete an MCP handshake over stdio and return the advertised tools."""
    process = subprocess.Popen(
        [interpreter, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        def send(message: dict) -> None:
            assert process.stdin is not None
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        def receive() -> dict:
            assert process.stdout is not None
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "server closed stdout without replying; it most likely failed "
                    "to import -- run the interpreter against src/mcp_server.py "
                    "directly to see the traceback"
                )
            return json.loads(line)

        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "check_mcp_server", "version": "1"},
                },
            }
        )
        info = receive()["result"]["serverInfo"]
        print(f"Server: {info['name']} {info['version']} via {interpreter}")

        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return receive()["result"]["tools"]
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the CT Segmentation MCP server starts and lists its tools."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to launch the server with (default: this one)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="seconds to wait for shutdown"
    )
    arguments = parser.parse_args()

    tools = list_tools(arguments.python, arguments.timeout)
    names = {tool["name"] for tool in tools}
    for name in sorted(names):
        print(f"  - {name}")

    missing = EXPECTED_TOOLS - names
    if missing:
        print(
            f"\nFAIL: {len(missing)} expected tool(s) missing: "
            f"{', '.join(sorted(missing))}.\nThe server is probably an older "
            "checkout; check the path in ~/.codex/config.toml.",
            file=sys.stderr,
        )
        return 1

    extra = names - EXPECTED_TOOLS
    if extra:
        print(f"\nNote: {len(extra)} tool(s) beyond the expected set: {', '.join(sorted(extra))}")
    print(f"\nOK: all {len(EXPECTED_TOOLS)} expected tools are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
