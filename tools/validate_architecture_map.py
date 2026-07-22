"""Validate architecture-map.json and its references using the standard library."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "architecture-map.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))

    require(data.get("schemaVersion") == "1.0.0", "Unsupported schemaVersion")
    require(data.get("generatedFrom") == "docs/multi_agent_driving_mvp_spec.md", "Unexpected source document")

    categories = data.get("categories", [])
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    flows = data.get("flows", [])
    phases = data.get("phases", [])

    category_ids = {category["id"] for category in categories}
    node_ids = {node["id"] for node in nodes}
    edge_ids = {edge["id"] for edge in edges}
    phase_ids = {phase["id"] for phase in phases}

    require(len(category_ids) == len(categories), "Duplicate category id")
    require(len(node_ids) == len(nodes), "Duplicate node id")
    require(len(edge_ids) == len(edges), "Duplicate edge id")
    require(len(phase_ids) == len(phases), "Duplicate phase id")

    for node in nodes:
        require(node["category"] in category_ids, f"Unknown category on {node['id']}")
        require(node["phase"] in phase_ids, f"Unknown phase on {node['id']}")
        require(node.get("status") in {"planned", "implemented", "partial"}, f"Invalid status on {node['id']}")
        require(bool(node.get("description")), f"Missing description on {node['id']}")
        require(bool(node.get("responsibilities")), f"Missing responsibilities on {node['id']}")
        require(bool(node.get("sourcePaths")), f"Missing source paths on {node['id']}")
        position = node.get("position", {})
        require(isinstance(position.get("x"), (int, float)), f"Missing x position on {node['id']}")
        require(isinstance(position.get("y"), (int, float)), f"Missing y position on {node['id']}")

    for edge in edges:
        require(edge["source"] in node_ids, f"Unknown edge source on {edge['id']}")
        require(edge["target"] in node_ids, f"Unknown edge target on {edge['id']}")
        require(edge["source"] != edge["target"], f"Self edge on {edge['id']}")
        require(bool(edge.get("contract")), f"Missing contract on {edge['id']}")

    for flow in flows:
        flow_id = flow.get("id")
        require(bool(flow_id), "Flow is missing id")
        require(bool(flow.get("description")), f"Missing description on flow {flow_id}")
        require(bool(flow.get("actor")), f"Missing actor on flow {flow_id}")
        require(bool(flow.get("trigger")), f"Missing trigger on flow {flow_id}")
        require(bool(flow.get("outcome")), f"Missing outcome on flow {flow_id}")
        require(bool(flow.get("evidence")), f"Missing evidence on flow {flow_id}")
        require(bool(flow.get("coverageNote")), f"Missing coverage note on flow {flow_id}")
        require(bool(flow.get("safetyChecks")), f"Missing safety checks on flow {flow_id}")
        path = flow.get("nodePath", [])
        require(len(path) >= 2, f"Flow is too short: {flow_id}")
        unknown = [node_id for node_id in path if node_id not in node_ids]
        require(not unknown, f"Unknown nodes in flow {flow_id}: {unknown}")
        unknown_focus_nodes = [node_id for node_id in flow.get("nodeIds", []) if node_id not in node_ids]
        require(not unknown_focus_nodes, f"Unknown focused nodes in flow {flow_id}: {unknown_focus_nodes}")
        unknown_flow_edges = [edge_id for edge_id in flow.get("edgeIds", []) if edge_id not in edge_ids]
        require(not unknown_flow_edges, f"Unknown focused edges in flow {flow_id}: {unknown_flow_edges}")

        stages = flow.get("stages", [])
        require(2 <= len(stages) <= 6, f"Unexpected stage count on flow {flow_id}")
        stage_ids = {stage.get("id") for stage in stages}
        require(None not in stage_ids and len(stage_ids) == len(stages), f"Invalid stage ids on flow {flow_id}")
        for stage in stages:
            require(bool(stage.get("label")), f"Missing stage label on flow {flow_id}")
            require(bool(stage.get("description")), f"Missing stage description on flow {flow_id}")
            require(bool(stage.get("backstage")), f"Missing backstage detail on flow {flow_id}")
            require(bool(stage.get("produces")), f"Missing stage output on flow {flow_id}")
            stage_nodes = stage.get("nodeIds", [])
            require(bool(stage_nodes), f"Stage has no modules on flow {flow_id}")
            unknown_stage_nodes = [node_id for node_id in stage_nodes if node_id not in node_ids]
            require(not unknown_stage_nodes, f"Unknown stage nodes in flow {flow_id}: {unknown_stage_nodes}")

    require((ROOT / data["generatedFrom"]).exists(), "Source specification is missing")
    require((ROOT / "architecture-map.html").exists(), "HTML map is missing")

    print(
        "architecture-map.json is valid: "
        f"{len(nodes)} nodes, {len(edges)} edges, {len(flows)} flows, {len(phases)} phases"
    )


if __name__ == "__main__":
    main()
