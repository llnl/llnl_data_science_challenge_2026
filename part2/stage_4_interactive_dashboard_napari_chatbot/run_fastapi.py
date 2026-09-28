"""FastAPI broker for the Napari viewer and the future Next.js dashboard.

The API deliberately keeps the CT volume out of the web process.  Napari owns
the TIFF and this service exposes the precomputed geometry and Stage 3
metadata, which keeps dashboard requests small and predictable.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTERLINES = REPO_ROOT / "part2/napari_visualizer/napari_centerlines.csv"
DEFAULT_EXPORT = REPO_ROOT / "part2/stage_3_defect_analysis/output/station_export_20260727T193004Z"
DEFAULT_DEFECT_BY_STRUT = DEFAULT_EXPORT / "defect_analysis_by_strut.csv"
DEFAULT_DEFECT_BY_STATION = DEFAULT_EXPORT / "defect_analysis_by_station.csv"
logger = logging.getLogger(__name__)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _json_value(value: Any) -> Any:
    """Convert Pandas/NumPy scalar values to JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_json_value(row) for row in frame.to_dict(orient="records")]


class DataStore:
    def __init__(self, centerlines: Path, defect_by_strut: Path, defect_by_station: Path):
        self.paths = {
            "centerlines": centerlines,
            "defect_by_strut": defect_by_strut,
            "defect_by_station": defect_by_station,
        }
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing dashboard data file(s): " + ", ".join(missing))

        self.centerlines = pd.read_csv(centerlines)
        self.defect_by_strut = pd.read_csv(defect_by_strut)
        self.defect_by_station = pd.read_csv(defect_by_station)
        for name, frame in (
            ("centerlines", self.centerlines),
            ("defect_by_strut", self.defect_by_strut),
            ("defect_by_station", self.defect_by_station),
        ):
            if "strut_id" not in frame.columns:
                raise ValueError(f"{name} must contain a strut_id column")

        self.centerlines["_strut_key"] = self.centerlines["strut_id"].astype(str)
        self.defect_by_strut["_strut_key"] = self.defect_by_strut["strut_id"].astype(str)
        self.defect_by_station["_strut_key"] = self.defect_by_station["strut_id"].astype(str)
        if self.centerlines["_strut_key"].duplicated().any():
            raise ValueError("centerlines must contain one row per strut_id")
        if self.defect_by_strut["_strut_key"].duplicated().any():
            raise ValueError("defect_by_strut must contain one row per strut_id")
        self.strut_keys = set(self.centerlines["_strut_key"])
        self.centerline_by_key = self.centerlines.set_index("_strut_key", drop=False)
        self.defect_by_key = self.defect_by_strut.set_index("_strut_key", drop=False)
        self.station_indices_by_key = self.defect_by_station.groupby("_strut_key").indices

        # Streamlit needs a complete, filterable defect table but not Napari
        # geometry. Serialize this once at startup rather than rebuilding
        # 18k records for every dashboard request.
        self.dashboard_summary = self.defect_by_strut.drop(columns="_strut_key").copy()
        self.dashboard_records = _records(self.dashboard_summary)

    @staticmethod
    def _value(row: pd.Series, *columns: str) -> Any:
        """Return the first non-null value available under the supplied aliases."""
        for column in columns:
            if column in row.index and pd.notna(row[column]):
                return _json_value(row[column])
        return None

    def _compact_strut_record(self, key: str) -> Dict[str, Any]:
        """Merge the useful inventory and defect fields for dashboard clients."""
        geometry = self.centerline_by_key.loc[key]
        defect = self.defect_by_key.loc[key] if key in self.defect_by_key.index else pd.Series(dtype=object)

        def value(*columns: str) -> Any:
            defect_value = self._value(defect, *columns)
            return defect_value if defect_value is not None else self._value(geometry, *columns)

        record = {
            "strut_id": self._value(geometry, "strut_id"),
            "classification": value("stage2_classification", "inventory_classification"),
            "primary_defect": value("primary_defect"),
            "needs_review": value("needs_review"),
            "expected_by_0point5_cad": value("expected_by_0point5_cad", "inventory_expected_by_0point5_cad"),
            "unit_cell_ids": value("unit_cell_ids", "inventory_unit_cell_ids"),
            "junction0_id": value("junction0_id", "inventory_junction0_id"),
            "junction1_id": value("junction1_id", "inventory_junction1_id"),
            "junction0_degree": value("junction0_degree", "inventory_junction0_degree"),
            "junction1_degree": value("junction1_degree", "inventory_junction1_degree"),
            "length_um": value("inventory_length_um", "length_um_from_centerline", "length_um"),
            "center_z_vox": self._value(geometry, "center_z_vox"),
            "center_y_vox": self._value(geometry, "center_y_vox"),
            "center_x_vox": self._value(geometry, "center_x_vox"),
            "start_z_vox": self._value(geometry, "start_z_vox"),
            "start_y_vox": self._value(geometry, "start_y_vox"),
            "start_x_vox": self._value(geometry, "start_x_vox"),
            "end_z_vox": self._value(geometry, "end_z_vox"),
            "end_y_vox": self._value(geometry, "end_y_vox"),
            "end_x_vox": self._value(geometry, "end_x_vox"),
        }
        record["defect_summary"] = (
            {column: _json_value(value) for column, value in defect.items() if column != "_strut_key"}
            if not defect.empty
            else {}
        )
        return record

    def summary(self) -> Dict[str, Any]:
        classification_column = (
            "stage2_classification"
            if "stage2_classification" in self.defect_by_strut.columns
            else "inventory_classification"
        )
        classifications = self.defect_by_strut[classification_column].value_counts().to_dict()
        defects = (
            self.defect_by_strut["primary_defect"].value_counts().to_dict()
            if "primary_defect" in self.defect_by_strut.columns
            else {}
        )
        unit_cell_column = (
            "unit_cell_ids" if "unit_cell_ids" in self.defect_by_strut.columns else "inventory_unit_cell_ids"
        )
        return {
            "strut_count": len(self.centerlines),
            "station_record_count": len(self.defect_by_station),
            "station_count_per_strut": int(self.defect_by_station.groupby("_strut_key").size().mode().iloc[0]),
            "stage2_classification_counts": _json_value(classifications),
            "primary_defect_counts": _json_value(defects),
            "unit_cell_count": int(self.defect_by_strut[unit_cell_column].astype(str).nunique()),
            "data_files": {key: str(path) for key, path in self.paths.items()},
        }

    def require_key(self, strut_id: int | str) -> str:
        key = str(strut_id)
        if key not in self.strut_keys:
            raise HTTPException(status_code=404, detail=f"Unknown strut_id: {strut_id}")
        return key

    def strut_detail(self, strut_id: int | str) -> Dict[str, Any]:
        key = self.require_key(strut_id)
        stations = self.station_frame(key)
        return {
            "strut_id": int(strut_id),
            "strut": self._compact_strut_record(key),
            "stations": _records(stations),
        }

    def station_frame(self, key: str) -> pd.DataFrame:
        indices = self.station_indices_by_key.get(key, [])
        return self.defect_by_station.iloc[indices].drop(columns="_strut_key")


def load_store() -> DataStore:
    return DataStore(
        _env_path("NAPARI_CENTERLINES_CSV", DEFAULT_CENTERLINES),
        _env_path("DEFECT_BY_STRUT_CSV", DEFAULT_DEFECT_BY_STRUT),
        _env_path("DEFECT_BY_STATION_CSV", DEFAULT_DEFECT_BY_STATION),
    )


class GlobalState(BaseModel):
    active_strut_id: Optional[int] = None
    active_strut_ids: List[int] = Field(default_factory=list)
    active_station_id: Optional[int] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class SelectionRequest(BaseModel):
    strut_ids: List[int] = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    active_strut_id: Optional[int] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send(self, websocket: WebSocket, message: Dict[str, Any]):
        await websocket.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        stale = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


app = FastAPI(title="CT Defect Analysis Dashboard Broker", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "DASHBOARD_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = load_store()
current_state = GlobalState()
manager = ConnectionManager()


@app.get("/")
def read_root():
    return {"status": "Central Broker Running", "active_clients": len(manager.active_connections)}


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": True}


@app.get("/frontend-config")
def frontend_config():
    return {
        "api_status": "ok",
        "websocket_path": "/ws",
        "capabilities": ["summary", "strut_listing", "strut_detail", "stations", "selection", "chat"],
        "summary": store.summary(),
    }


@app.get("/summary")
def get_summary():
    return store.summary()


@app.get("/state")
def get_state():
    return current_state.model_dump() if hasattr(current_state, "model_dump") else current_state.dict()


@app.get("/dashboard/struts")
def dashboard_struts():
    """Return the lightweight, complete defect table used by Streamlit."""
    return {"count": len(store.dashboard_records), "items": store.dashboard_records}


@app.get("/struts")
def list_struts(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    classification: Optional[str] = None,
    primary_defect: Optional[str] = None,
    needs_review: Optional[bool] = None,
    unit_cell_id: Optional[str] = None,
    search: Optional[str] = None,
):
    frame = store.defect_by_strut
    if classification and "stage2_classification" in frame.columns:
        frame = frame[frame["stage2_classification"].astype(str).str.casefold() == classification.casefold()]
    elif classification:
        centerline_keys = set(
            store.centerlines.loc[
                store.centerlines["inventory_classification"].astype(str).str.casefold() == classification.casefold(),
                "_strut_key",
            ]
        )
        frame = frame[frame["_strut_key"].isin(centerline_keys)]
    if primary_defect and "primary_defect" in frame.columns:
        frame = frame[frame["primary_defect"].astype(str).str.casefold() == primary_defect.casefold()]
    if needs_review is not None and "needs_review" in frame.columns:
        review_values = frame["needs_review"].astype(str).str.casefold().isin({"true", "1", "yes"})
        frame = frame[review_values == needs_review]
    if unit_cell_id and "unit_cell_ids" in frame.columns:
        frame = frame[frame["unit_cell_ids"].astype(str).str.split(",").apply(lambda values: unit_cell_id in values)]
    if search:
        frame = frame[frame["_strut_key"].str.contains(search, case=False, regex=False)]
    total = len(frame)
    result_keys = frame.iloc[offset : offset + limit]["_strut_key"].tolist()
    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "items": [store._compact_strut_record(key) for key in result_keys],
    }


@app.get("/struts/{strut_id}")
def get_strut(strut_id: int):
    return store.strut_detail(strut_id)


@app.get("/struts/{strut_id}/stations")
def get_stations(strut_id: int):
    key = store.require_key(strut_id)
    frame = store.station_frame(key)
    return {"strut_id": strut_id, "count": len(frame), "items": _records(frame)}


async def publish_selection(strut_ids: List[int]) -> Dict[str, Any]:
    if not strut_ids:
        raise HTTPException(status_code=422, detail="At least one strut_id is required")
    for strut_id in strut_ids:
        store.require_key(strut_id)
    current_state.active_strut_ids = list(dict.fromkeys(strut_ids))
    current_state.active_strut_id = current_state.active_strut_ids[0]
    event = {"event_type": "STRUTS_SELECTED", "data": {"strut_ids": current_state.active_strut_ids}}
    await manager.broadcast(event)
    return {"status": "success", **event["data"]}


@app.post("/select_strut")
async def select_strut(strut_id: int):
    return await publish_selection([strut_id])


@app.post("/select_struts")
async def select_struts(request: SelectionRequest):
    return await publish_selection(request.strut_ids)


@app.post("/chat")
async def chat(request: ChatRequest):
    """HTTP chat endpoint for clients that do not maintain a WebSocket."""
    result = _chat_response(request.message, request.active_strut_id)
    selected_ids = result.pop("select_strut_ids", None)
    if selected_ids:
        await publish_selection(selected_ids)
    return {"message_id": str(uuid4()), **result}


def _chat_response(message: str, active_strut_id: Optional[int] = None) -> Dict[str, Any]:
    text = message.strip()
    lower = text.casefold()
    normalized_text = re.sub(r"[\s_-]+", "", lower)
    ids = [int(value) for value in re.findall(r"\b(?:strut\s*)?(\d+)\b", lower)]
    if "select" in lower and ids:
        valid = [value for value in ids if str(value) in store.strut_keys]
        if valid:
            return {"reply": f"Selecting strut(s): {', '.join(map(str, valid))}.", "select_strut_ids": valid, "references": []}

    if ids and ("station" in lower or "detail" in lower or "inspect" in lower or "information" in lower):
        detail = store.strut_detail(ids[0])
        strut = detail["strut"]
        return {
            "reply": (
                f"Strut {ids[0]} is classified as {strut.get('classification', 'unknown')} "
                f"with primary defect {strut.get('primary_defect', 'unknown')}. "
                f"It has {len(detail['stations'])} station measurements."
            ),
            "strut": detail,
            "references": [f"struts/{ids[0]}", f"struts/{ids[0]}/stations"],
        }

    requested_strut_id = ids[0] if ids else active_strut_id
    if requested_strut_id is not None and any(word in lower for word in ("deviation", "offset", "largest", "where")):
        detail = store.strut_detail(requested_strut_id)
        stations = pd.DataFrame(detail["stations"])
        offset_column = next(
            (
                column
                for column in ("centroid_offset_um", "centroid_offset", "centerline_offset_um")
                if column in stations.columns
            ),
            None,
        )
        if offset_column is not None:
            offsets = pd.to_numeric(stations[offset_column], errors="coerce").dropna()
            if not offsets.empty:
                max_index = offsets.idxmax()
                position = pd.to_numeric(stations.loc[max_index].get("position_fraction"), errors="coerce")
                if pd.notna(position):
                    return {
                        "reply": (
                            f"For strut {requested_strut_id}, the largest uploaded centerline offset is "
                            f"{float(offsets.loc[max_index]):g} at position_fraction {float(position):.4g}."
                        ),
                        "strut_id": requested_strut_id,
                        "max_offset_value": float(offsets.loc[max_index]),
                        "max_offset_position_fraction": float(position),
                        "references": [f"/struts/{requested_strut_id}/stations"],
                    }
        return {
            "reply": f"The station data does not provide a usable centerline-offset series for strut {requested_strut_id}.",
            "strut_id": requested_strut_id,
            "references": [f"/struts/{requested_strut_id}/stations"],
        }

    if "review" in lower:
        review_values = store.defect_by_strut["needs_review"].astype(str).str.casefold().isin({"true", "1", "yes"})
        frame = store.defect_by_strut[review_values]
        ids_for_review = [int(value) for value in frame["strut_id"].tolist()]
        return {
            "reply": f"{len(ids_for_review)} struts are marked as needing review.",
            "strut_ids": ids_for_review[:100],
            "total": len(ids_for_review),
            "references": ["/struts?needs_review=true"],
        }

    defect_names = sorted(store.defect_by_strut["primary_defect"].dropna().astype(str), key=str.casefold)
    classification_column = "stage2_classification"
    classification_names = sorted(
        store.defect_by_strut[classification_column].dropna().astype(str), key=str.casefold
    )

    def normalized_label(value: str) -> str:
        """Make display labels match natural spacing and punctuation variants."""
        return re.sub(r"[\s_-]+", "", value.casefold())

    def matching_name(names: List[str]) -> Optional[str]:
        return next((name for name in names if normalized_label(name) in normalized_text), None)

    requested_defect = matching_name(defect_names)
    requested_classification = matching_name(classification_names)
    explicitly_defect = "primary defect" in lower
    explicitly_classification = "stage 2" in lower or "stage2" in lower or "classification" in lower

    if explicitly_defect:
        category_field, category_value = "primary_defect", requested_defect
    elif explicitly_classification:
        category_field, category_value = classification_column, requested_classification
    else:
        # A bare value that could be in either column (such as Nominal) means
        # classification; values unique to primary_defect (such as Bent) still
        # resolve naturally to their defect row.
        category_field, category_value = (
            (classification_column, requested_classification)
            if requested_classification
            else ("primary_defect", requested_defect)
        )

    list_intent = any(token in lower for token in ("which", "list", "show", "find"))
    missing_word = bool(re.search(r"\bmissing\b", re.sub(r"[_-]", " ", lower)))
    if missing_word and not category_value:
        missing_counts = {
            name: int((store.defect_by_strut["primary_defect"].astype(str) == name).sum())
            for name in ("Missing_Intentional", "Missing_Unintentional")
        }
        return {
            "reply": (
                "Please specify whether you mean Missing_Intentional "
                f"({missing_counts['Missing_Intentional']} struts) or "
                f"Missing_Unintentional ({missing_counts['Missing_Unintentional']} struts)."
            ),
            "missing_subtypes": missing_counts,
            "references": [
                "/struts?primary_defect=Missing_Intentional",
                "/struts?primary_defect=Missing_Unintentional",
            ],
        }

    if category_value and list_intent:
        frame = store.defect_by_strut[
            store.defect_by_strut[category_field].astype(str).str.casefold() == category_value.casefold()
        ]
        strut_ids = [int(value) for value in frame["strut_id"].tolist()]
        label = "primary defect" if category_field == "primary_defect" else "Stage 2 classification"
        shown_ids = strut_ids[:100]
        return {
            "reply": (
                f"There are {len(strut_ids)} struts with {label} {category_value}. "
                f"Showing the first {len(shown_ids)}: {', '.join(map(str, shown_ids))}."
            ),
            "strut_ids": shown_ids,
            "total": len(strut_ids),
            "field": category_field,
            "value": category_value,
            "references": [f"/struts?{'primary_defect' if category_field == 'primary_defect' else 'classification'}={category_value}"],
        }

    if requested_defect and ("how many" in lower or "count" in lower or "summary" in lower):
        count = int((store.defect_by_strut["primary_defect"].astype(str) == requested_defect).sum())
        return {
            "reply": f"There are {count} struts with primary defect {requested_defect}.",
            "defect": requested_defect,
            "count": count,
            "references": [f"/struts?primary_defect={requested_defect}"],
        }

    if "how many" in lower or "count" in lower or "summary" in lower or "defect" in lower:
        summary = store.summary()
        return {
            "reply": (
                f"The dataset contains {summary['strut_count']} struts. Primary defects: "
                + ", ".join(f"{key}: {value}" for key, value in summary["primary_defect_counts"].items())
                + "."
            ),
            "summary": summary,
            "references": ["summary"],
        }

    return {
        "reply": "I can summarize defects, inspect a strut or its stations, list review candidates, and select struts.",
        "references": [],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        state = get_state()
        await manager.send(websocket, {"event_type": "INIT_STATE", "data": state})
        while True:
            try:
                incoming = await websocket.receive_json()
                if not isinstance(incoming, dict):
                    raise ValueError("WebSocket event must be a JSON object")
                event_type = incoming.get("event_type", incoming.get("type", "")).upper()
                data = incoming.get("data", incoming)
                if not isinstance(data, dict):
                    raise ValueError("WebSocket event data must be a JSON object")
                request_id = str(data.get("request_id", "")) or None
                if event_type in {"CHAT_REQUEST", "CHAT"}:
                    request = ChatRequest(message=str(data.get("message", "")))
                    result = _chat_response(request.message)
                    selected_ids = result.pop("select_strut_ids", None)
                    if selected_ids:
                        await publish_selection(selected_ids)
                    await manager.send(
                        websocket,
                        {
                            "event_type": "CHAT_RESPONSE",
                            "data": {"message_id": str(uuid4()), "request_id": request_id, **result},
                        },
                    )
                elif event_type in {"STRUT_SELECTED", "STRUTS_SELECTED"}:
                    requested = data.get("strut_ids", [data.get("strut_id")])
                    await publish_selection([int(value) for value in requested if value is not None])
                else:
                    raise ValueError(f"Unknown event type: {event_type or '(empty)'}")
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.exception("WebSocket event failed from %s: %s", websocket.client, exc)
                try:
                    await manager.send(
                        websocket,
                        {"event_type": "ERROR", "data": {"message": f"Unable to process request: {exc}"}},
                    )
                except Exception:
                    logger.info("WebSocket error response could not be delivered to %s", websocket.client)
                    raise WebSocketDisconnect(code=1011)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Unhandled WebSocket failure for %s", websocket.client)
    finally:
        manager.disconnect(websocket)
