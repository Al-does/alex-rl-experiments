#!/usr/bin/env python3
"""Fetch Cursor Cloud Agent conversations into .cursor/cloud-transcripts/."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

API_BASE = "https://api.cursor.com"
DEFAULT_OUT_DIR = Path(".cursor/cloud-transcripts")


class CursorApiError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    env_key = __import__("os").environ.get("CURSOR_API_KEY", "").strip()
    if env_key:
        return env_key
    raise SystemExit(
        "Missing CURSOR_API_KEY. Create one at https://cursor.com/dashboard/api "
        "and add `export CURSOR_API_KEY=\"...\"` to .env.local, then run "
        "`source scripts/load_env.sh`."
    )


def request_json(
    api_key: str,
    method: str,
    path: str,
    *,
    query: dict[str, str] | None = None,
) -> Any:
    url = f"{API_BASE}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    auth = base64.b64encode(f"{api_key}:".encode()).decode()
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise CursorApiError(exc.code, body) from exc


def list_agents_v1(api_key: str, *, limit: int, cursor: str | None) -> dict[str, Any]:
    query: dict[str, str] = {"limit": str(min(limit, 100))}
    if cursor:
        query["cursor"] = cursor
    return request_json(api_key, "GET", "/v1/agents", query=query)


def list_agents_v0(api_key: str, *, limit: int, cursor: str | None) -> dict[str, Any]:
    query: dict[str, str] = {"limit": str(min(limit, 100))}
    if cursor:
        query["cursor"] = cursor
    return request_json(api_key, "GET", "/v0/agents", query=query)


def get_agent_v0(api_key: str, agent_id: str) -> dict[str, Any]:
    return request_json(api_key, "GET", f"/v0/agents/{urllib.parse.quote(agent_id, safe='')}")


def get_conversation(api_key: str, agent_id: str) -> dict[str, Any]:
    return request_json(
        api_key,
        "GET",
        f"/v0/agents/{urllib.parse.quote(agent_id, safe='')}/conversation",
    )


def normalize_agent_id(agent_id: str) -> str:
    return agent_id.strip()


def agent_record_from_v1(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "url": item.get("url"),
        "createdAt": item.get("createdAt"),
        "updatedAt": item.get("updatedAt"),
        "latestRunId": item.get("latestRunId"),
    }


def agent_record_from_v0(item: dict[str, Any]) -> dict[str, Any]:
    source = item.get("source") or {}
    target = item.get("target") or {}
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "url": target.get("url"),
        "createdAt": item.get("createdAt"),
        "repository": source.get("repository"),
        "ref": source.get("ref"),
        "branchName": target.get("branchName"),
        "prUrl": target.get("prUrl"),
        "summary": item.get("summary"),
    }


def list_recent_agents(api_key: str, *, limit: int) -> list[dict[str, Any]]:
    try:
        payload = list_agents_v1(api_key, limit=limit, cursor=None)
        return [agent_record_from_v1(item) for item in payload.get("items", [])]
    except CursorApiError:
        payload = list_agents_v0(api_key, limit=limit, cursor=None)
        return [agent_record_from_v0(item) for item in payload.get("agents", [])]


def message_to_jsonl_line(message: dict[str, Any]) -> dict[str, Any]:
    role = "user" if message.get("type") == "user_message" else "assistant"
    text = message.get("text", "")
    return {
        "role": role,
        "message": {"content": [{"type": "text", "text": text}]},
        "cloud_message_id": message.get("id"),
        "cloud_message_type": message.get("type"),
    }


def write_transcript(
    out_dir: Path,
    agent_id: str,
    *,
    conversation: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w\-]", "_", agent_id)
    jsonl_path = out_dir / f"{safe_id}.jsonl"
    meta_path = out_dir / f"{safe_id}.meta.json"

    lines = [message_to_jsonl_line(msg) for msg in conversation.get("messages", [])]
    jsonl_path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )

    meta = {
        "agent_id": agent_id,
        "fetched_at": datetime.now(UTC).isoformat(),
        "message_count": len(lines),
        "transcript_path": str(jsonl_path.relative_to(repo_root())),
        **metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return jsonl_path


def update_index(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    index_path = out_dir / "index.json"
    existing: dict[str, Any] = {"agents": []}
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
    by_id = {item["agent_id"]: item for item in existing.get("agents", []) if item.get("agent_id")}
    for entry in entries:
        by_id[entry["agent_id"]] = entry
    merged = sorted(by_id.values(), key=lambda item: item.get("updatedAt") or item.get("createdAt") or "", reverse=True)
    index_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "agents": merged,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def fetch_agent(api_key: str, agent_id: str, out_dir: Path) -> Path:
    agent_id = normalize_agent_id(agent_id)
    conversation = get_conversation(api_key, agent_id)
    metadata: dict[str, Any] = {}
    try:
        metadata = agent_record_from_v0(get_agent_v0(api_key, agent_id))
    except CursorApiError as exc:
        metadata = {"metadata_error": str(exc)}
    path = write_transcript(out_dir, agent_id, conversation=conversation, metadata=metadata)
    update_index(
        out_dir,
        [
            {
                "agent_id": agent_id,
                "name": metadata.get("name"),
                "status": metadata.get("status"),
                "repository": metadata.get("repository"),
                "branchName": metadata.get("branchName"),
                "prUrl": metadata.get("prUrl"),
                "summary": metadata.get("summary"),
                "createdAt": metadata.get("createdAt"),
                "updatedAt": metadata.get("updatedAt"),
                "transcript_path": str(path.relative_to(repo_root())),
            }
        ],
    )
    return path


def cmd_list(api_key: str, limit: int) -> int:
    agents = list_recent_agents(api_key, limit=limit)
    if not agents:
        print("No cloud agents returned.")
        return 0
    for agent in agents:
        repo = agent.get("repository") or ""
        pr = agent.get("prUrl") or ""
        print(
            f"{agent.get('id')}\t{agent.get('status')}\t{agent.get('createdAt')}\t"
            f"{agent.get('name')}\t{repo}\t{pr}"
        )
    return 0


def cmd_fetch(api_key: str, agent_ids: list[str], out_dir: Path) -> int:
    for agent_id in agent_ids:
        path = fetch_agent(api_key, agent_id, out_dir)
        print(f"Wrote {path}")
    return 0


def cmd_fetch_recent(api_key: str, limit: int, out_dir: Path) -> int:
    agents = list_recent_agents(api_key, limit=limit)
    if not agents:
        print("No cloud agents returned.")
        return 0
    for agent in agents:
        agent_id = agent.get("id")
        if not agent_id:
            continue
        path = fetch_agent(api_key, agent_id, out_dir)
        print(f"Wrote {path}")
    return 0


def cmd_search(out_dir: Path, query: str) -> int:
    if not out_dir.exists():
        print(f"No cached transcripts in {out_dir}. Run fetch-recent first.")
        return 1
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits = 0
    for path in sorted(out_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                print(f"{path}:{line_no}")
                hits += 1
    print(f"{hits} matching line(s)")
    return 0 if hits else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", help="Override CURSOR_API_KEY")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List recent cloud agents")
    list_cmd.add_argument("--limit", type=int, default=20)

    fetch_cmd = sub.add_parser("fetch", help="Fetch one or more agent conversations by id")
    fetch_cmd.add_argument("agent_ids", nargs="+")

    recent_cmd = sub.add_parser("fetch-recent", help="Fetch conversations for recent cloud agents")
    recent_cmd.add_argument("--limit", type=int, default=20)

    search_cmd = sub.add_parser("search", help="Search cached local transcript files")
    search_cmd.add_argument("query")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = load_api_key(args.api_key)
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir

    if args.command == "list":
        return cmd_list(api_key, args.limit)
    if args.command == "fetch":
        return cmd_fetch(api_key, args.agent_ids, out_dir)
    if args.command == "fetch-recent":
        return cmd_fetch_recent(api_key, args.limit, out_dir)
    if args.command == "search":
        return cmd_search(out_dir, args.query)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
