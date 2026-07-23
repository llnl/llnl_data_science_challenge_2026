"""Publish the 9x9x9 octet skeleton node-degree histogram to Crosshair."""

from __future__ import annotations

import asyncio
from pathlib import Path


HISTOGRAM_FILE = Path(
    "outputs/9x9x9_octet_skeleton_graph/node_degree_histogram.csv"
)
VIEW_NAME = "9x9x9-octet-skeleton-graph"
PANEL_ID = "octet-skeleton-node-degree-histogram"


async def publish_degree_histogram(histogram_file: Path = HISTOGRAM_FILE) -> str:
    """Create or update the Crosshair tab using a saved degree histogram CSV."""
    if not histogram_file.is_file():
        raise FileNotFoundError(
            f"Degree histogram not found: {histogram_file}. Create the skeleton graph first."
        )

    from crosshair import client

    url = await client.ensure()
    await client.call("create_view", name=VIEW_NAME, rows=1, cols=1)
    await client.call(
        "upsert_panel",
        view=VIEW_NAME,
        panel_id=PANEL_ID,
        title="9×9×9 Octet Lattice — Skeleton Node Degree",
        type="plotly",
        spec={
            "data": [
                {
                    "type": "bar",
                    "name": "Skeleton voxels",
                    "x": {"$ref": {"file": str(histogram_file), "column": "degree"}},
                    "y": {
                        "$ref": {"file": str(histogram_file), "column": "node_count"}
                    },
                    "hovertemplate": "Degree %{x}<br>Nodes %{y:,}<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"title": {"text": "Number of edges per node (degree)"}},
                "yaxis": {
                    "title": {"text": "Number of skeleton nodes (log scale)"},
                    "type": "log",
                },
                "margin": {"l": 70, "r": 20, "t": 20, "b": 70},
            },
        },
        row=1,
        col=1,
        base_dir=str(Path.cwd()),
    )
    await client.call(
        "add_note",
        text=(
            "Added a graph-analysis tab. Each bar is the number of skeleton-voxel "
            "nodes with the displayed count of Skan neighbor-graph edges."
        ),
    )
    return url


def main() -> None:
    """Publish the current 9x9x9 octet degree histogram."""
    print(asyncio.run(publish_degree_histogram()))


if __name__ == "__main__":
    main()
