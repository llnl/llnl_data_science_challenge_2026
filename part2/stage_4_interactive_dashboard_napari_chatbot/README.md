# FastAPI Dashboard Broker

The broker loads the precomputed Napari centerlines and Stage 3 defect
analysis exports. It does not load the CT volume; the Napari process owns that
large file.

Run from the repository root in Windows PowerShell:

```powershell
uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app --host 127.0.0.1 --port 8000
```

For active code development, add `--reload`. Reloading intentionally restarts
the broker and disconnects active WebSocket clients, so leave it off for a
stable Napari/chat session.

Useful endpoints are `/frontend-config`, `/summary`, `/struts`,
`/struts/{strut_id}`, `/struts/{strut_id}/stations`, `/state`, and
`POST /select_struts` with `{"strut_ids": [1, 2]}`. Streamlit uses `GET
/dashboard/struts` for its one-time defect-summary load, `GET /state` for
polling, and `POST /chat` for broker-backed chat. The WebSocket endpoint is
`/ws`.

Start the Streamlit client from the repository root after starting FastAPI:

```powershell
streamlit run part2/stage_4_interactive_dashboard_napari_chatbot/dashboard/app_defect_dashboard.py
```

The client defaults to `http://127.0.0.1:8000`. Set `DASHBOARD_API_URL` to
use another broker URL.

Next.js should use REST for initial data and connect to `/ws` for
`STRUTS_SELECTED`, `CHAT_RESPONSE`, and `ERROR` events. Send chat messages as:

```json
{
  "event_type": "CHAT_REQUEST",
  "data": {"message": "How many struts need review?"}
}
```

The Napari viewer can subscribe to dashboard selections by passing
`--api-url http://localhost:8000` (or setting `NAPARI_API_URL`). Standalone
Napari use remains unchanged when that option is omitted.

The default data paths can be overridden with `NAPARI_CENTERLINES_CSV`,
`DEFECT_BY_STRUT_CSV`, and `DEFECT_BY_STATION_CSV`. Local Next.js origins are
allowed by default; set `DASHBOARD_ALLOWED_ORIGINS` to a comma-separated list
for another deployment.

## Terminal chatbot

With the broker running, start the terminal client from the repository root:

```powershell
python .\part2\stage_4_interactive_dashboard_napari_chatbot\run_agent_cli.py
```

The client prints examples when it starts. Type `help` to show them again,
or use prompts such as `How many struts need review?`, `Inspect strut 42`,
`Which struts are classified as Bent?`,
`List struts with Stage 2 classification Missing_Intentional`,
`Which struts are Missing Intentional?`,
`Summarize the defects`, or `Select struts 12 and 18`. Listing prompts return
up to 100 IDs plus the total matching count. Selection prompts are broadcast
to connected Napari clients. This client talks directly to FastAPI;
it does not use Ollama or `OPENAI_API_KEY`. To use a different broker URL, set
`DASHBOARD_WEBSOCKET_URL`, for example:

```powershell
$env:DASHBOARD_WEBSOCKET_URL = "ws://127.0.0.1:8000/ws"
python .\part2\stage_4_interactive_dashboard_napari_chatbot\run_agent_cli.py
```
