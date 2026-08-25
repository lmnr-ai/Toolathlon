#!/usr/bin/env python3
"""Duplicate a Notion template subpage using only the REST API.

Drop-in replacement for `notion_page_duplicator`'s CLI. The original has two
paths and both need a credential the integration key cannot supply: the default
one drives Notion's hosted MCP server through `mcp-remote`, which needs the
interactive OAuth state that `global_preparation/special_setup_notion_official`
writes to `configs/.mcp-auth`, and the fallback one drives a real browser with
the session cookies in `configs/notion_state.json`. Without either, `mcp-remote`
sits waiting for a callback that never arrives and the preprocess step hangs
until the task times out -- silently, because its output is only read once the
subprocess exits.

Notion's REST API has no "duplicate page" call, so this walks the template and
rebuilds it: blocks, inline databases, database schemas and database rows. That
covers the template pages under `Notion Source Page`, whose databases use only
scalar property types (no relations, rollups or formulas, which the API cannot
faithfully recreate anyway).

Two properties cannot round-trip and are reported by `--verify` rather than
hidden:

  * `status` properties cannot be created through the API at all, so they become
    `select` properties with the same options. The values survive; the
    to-do/in-progress/done grouping does not.
  * `link_preview` blocks cannot be created through the API, so they become
    `bookmark` blocks pointing at the same URL.

Usage mirrors the module it replaces:

    python -m utils.app_specific.notion.notion_rest_duplicator \
        --source-parent <url> --child-name "Oil Price" \
        --target-parent <url> --notion-key <secret> --output-file <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# Notion's published limit is three requests per second averaged over a burst.
# Staying just under it is far cheaper than eating 429s on an 87-row database.
MIN_INTERVAL = 0.34

# Fields the API returns but rejects on write.
READ_ONLY_BLOCK_FIELDS = {
    "object", "id", "parent", "created_time", "last_edited_time", "created_by",
    "last_edited_by", "has_children", "archived", "in_trash", "request_id",
}
# Blocks whose body is a file reference rather than rich text.
MEDIA_BLOCKS = {"image", "video", "audio", "pdf", "file"}
# Property values that are computed by Notion and cannot be written.
COMPUTED_PROPERTY_TYPES = {
    "formula", "rollup", "created_time", "created_by", "last_edited_time",
    "last_edited_by", "unique_id", "button", "relation",
}


class NotionError(RuntimeError):
    pass


class Notion:
    """Throttled Notion REST client with 429 handling."""

    def __init__(self, token: str) -> None:
        self.token = token
        self._last_call = 0.0

    def request(self, method: str, path: str, body: dict | None = None,
                attempts: int = 5) -> dict:
        wait = MIN_INTERVAL - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            API + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            retryable = exc.code == 429 or exc.code >= 500
            if retryable and attempts > 1:
                delay = float(exc.headers.get("Retry-After") or 1.0)
                time.sleep(delay)
                return self.request(method, path, body, attempts - 1)
            raise NotionError(f"{method} {path} -> {exc.code}: {detail[:600]}") from None
        finally:
            self._last_call = time.monotonic()

    def paged(self, method: str, path: str, body: dict | None = None) -> list[dict]:
        """Collect every page of a paginated endpoint."""
        out: list[dict] = []
        cursor: str | None = None
        while True:
            if method == "GET":
                sep = "&" if "?" in path else "?"
                page = self.request("GET", f"{path}{sep}page_size=100"
                                    + (f"&start_cursor={cursor}" if cursor else ""))
            else:
                payload = dict(body or {}, page_size=100)
                if cursor:
                    payload["start_cursor"] = cursor
                page = self.request(method, path, payload)
            out.extend(page["results"])
            if not page.get("has_more"):
                return out
            cursor = page["next_cursor"]

    def children(self, block_id: str) -> list[dict]:
        return self.paged("GET", f"/blocks/{block_id}/children")


def page_id_from_url(url: str) -> str:
    """Accept a Notion URL or a bare id, and return the dashed uuid form."""
    raw = urlparse(url).path.split("/")[-1].split("-")[-1] if "/" in url else url
    raw = raw.replace("-", "")
    if len(raw) != 32:
        raise ValueError(f"cannot read a page id out of {url!r}")
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def strip_ids(value: Any) -> Any:
    """Drop the nulls Notion sends on read but rejects on write.

    Every optional field comes back explicitly null -- `paragraph.icon`,
    `text.link`, `rich_text[].href` -- and the create endpoints validate those
    as "should be an object or `undefined`". Absent means absent.
    """
    if isinstance(value, list):
        return [strip_ids(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_ids(v) for k, v in value.items() if v is not None}
    return value


def strip_option_ids(options: list[dict]) -> list[dict]:
    """Drop the server ids from select/status options.

    Options are echoed back with the source database's ids; the target database
    does not know them, so Notion rejects the schema rather than creating the
    options by name.
    """
    return [{k: v for k, v in o.items() if k != "id"} for o in options]


def externalize(obj: dict | None) -> dict | None:
    """Rewrite an uploaded-file reference as an external one.

    Anything uploaded to the workspace -- a callout icon, an image block, a page
    cover -- reads back as `{"type": "file", "file": {"url": <signed s3 url>}}`,
    and the create endpoints reject that shape outright: they accept only
    `external`, `emoji`, `custom_emoji` or `file_upload`. Re-pointing at the
    signed URL keeps the block and its position; the link itself expires within
    the hour, which outlasts a task run but not the page.
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("type") not in ("file", "file_upload"):
        return obj
    url = (obj.get(obj["type"]) or {}).get("url")
    return {"type": "external", "external": {"url": url}} if url else None


class Duplicator:
    def __init__(self, client: Notion, notes: list[str]) -> None:
        self.client = client
        self.notes = notes
        # the page currently being filled, used when a block cannot host a
        # database and the copy has to fall back to the enclosing page
        self.page_id: str | None = None

    # ---------- blocks ----------

    def block_payload(self, block: dict) -> dict | None:
        """Turn a fetched block into one the create endpoints will accept."""
        kind = block["type"]
        if kind in ("child_page", "child_database"):
            return None  # handled by the caller, which needs the new parent id
        if kind == "unsupported":
            self.notes.append("skipped an 'unsupported' block (not readable via API)")
            return None
        if kind == "link_preview":
            # Not creatable; a bookmark keeps the URL reachable from the page.
            self.notes.append("link_preview -> bookmark (link_preview is not creatable)")
            return {"type": "bookmark",
                    "bookmark": {"url": block["link_preview"]["url"]}}

        body = strip_ids(block[kind])

        if isinstance(body.get("icon"), dict):
            icon = externalize(body["icon"])
            if icon is None:
                body.pop("icon")
            else:
                if icon is not body["icon"]:
                    self.notes.append(f"{kind} icon: upload -> external URL")
                body["icon"] = icon
        if kind in MEDIA_BLOCKS and body.get("type") in ("file", "file_upload"):
            external = externalize(body)
            if external is None:
                self.notes.append(f"skipped {kind} block with an unreadable upload")
                return None
            self.notes.append(f"{kind}: upload -> external URL (signed link expires)")
            body = {k: v for k, v in body.items() if k in ("caption", "name")} | external

        if kind == "table":
            rows = [self.block_payload(r) for r in self.client.children(block["id"])]
            body = dict(body, children=[r for r in rows if r])
        return {"type": kind, kind: body}

    def copy_column_list(self, src_block: dict, dst_id: str) -> None:
        """Recreate a column layout under `dst_id`.

        Columns cannot be created empty and a column_list cannot be appended to
        after the fact, so each column is seeded with an empty paragraph, filled
        by the ordinary recursive copy, and then unseeded. Building the columns'
        contents inline instead would silently drop any database inside one.
        """
        src_columns = self.client.children(src_block["id"])
        seed = {"type": "paragraph", "paragraph": {"rich_text": []}}
        created = self.client.request("PATCH", f"/blocks/{dst_id}/children", {
            "children": [{
                "type": "column_list",
                "column_list": {"children": [
                    {"type": "column", "column": dict(
                        {k: v for k, v in strip_ids(c["column"]).items()
                         if k == "width_ratio"},
                        children=[seed],
                    )}
                    for c in src_columns
                ]},
            }],
        })["results"][0]

        for src_col, dst_col in zip(src_columns, self.client.children(created["id"])):
            self.copy_children(src_col["id"], dst_col["id"])
            planted = self.client.children(dst_col["id"])
            # a column whose only content was hoisted out has to keep the seed:
            # Notion drops an empty column and renumbers the layout
            if len(planted) > 1:
                self.client.request("DELETE", f"/blocks/{planted[0]['id']}")

    def copy_children(self, src_id: str, dst_id: str) -> None:
        """Recreate every child of `src_id` under `dst_id`, in order."""
        batch: list[dict] = []
        # A block whose children go in a second pass, paired with its source.
        deferred: list[tuple[dict, int]] = []

        def flush() -> list[dict]:
            if not batch:
                return []
            created = self.client.request(
                "PATCH", f"/blocks/{dst_id}/children", {"children": batch}
            )["results"]
            batch.clear()
            return created

        for child in self.client.children(src_id):
            kind = child["type"]
            if kind == "child_page":
                created = flush()
                self._resolve(deferred, created)
                self.copy_page(child["id"], dst_id, parent_is_page=True)
                continue
            if kind == "child_database":
                created = flush()
                self._resolve(deferred, created)
                try:
                    self.copy_database(child["id"], dst_id)
                except NotionError as exc:
                    # A linked view reads back as a child_database the bot has
                    # no data source for. Nothing can be done about it, and it
                    # must not cost us the blocks that follow it.
                    if "does not contain any data sources" not in str(exc):
                        raise
                    self.notes.append("skipped a linked database view "
                                      "(not readable through the API)")
                continue
            if kind == "column_list":
                created = flush()
                self._resolve(deferred, created)
                self.copy_column_list(child, dst_id)
                continue

            payload = self.block_payload(child)
            if payload is None:
                continue
            batch.append(payload)
            # a table already carries its rows inline
            if child.get("has_children") and kind != "table":
                deferred.append((child, len(batch) - 1))
            if len(batch) == 100:
                created = flush()
                self._resolve(deferred, created)

        created = flush()
        self._resolve(deferred, created)

    def _resolve(self, deferred: list[tuple[dict, int]], created: list[dict]) -> None:
        for source, index in deferred:
            self.copy_children(source["id"], created[index]["id"])
        deferred.clear()

    # ---------- pages ----------

    def copy_page(self, src_page_id: str, dst_parent_id: str,
                  title: str | None = None, parent_is_page: bool = True) -> str:
        src = self.client.request("GET", f"/pages/{src_page_id}")
        title_prop = next(
            (v for v in src["properties"].values() if v["type"] == "title"), None
        )
        rich_title = strip_ids(title_prop["title"]) if title_prop else []
        if title is not None:
            rich_title = [{"type": "text", "text": {"content": title}}]

        body: dict[str, Any] = {
            "parent": {"page_id" if parent_is_page else "database_id": dst_parent_id},
            "properties": {"title": rich_title},
        }
        for key in ("icon", "cover"):
            decoration = externalize(src.get(key))
            if decoration:
                body[key] = decoration

        new_page = self.client.request("POST", "/pages", body)
        enclosing, self.page_id = self.page_id, new_page["id"]
        try:
            self.copy_children(src_page_id, new_page["id"])
        finally:
            self.page_id = enclosing
        return new_page["id"]

    # ---------- databases ----------

    def schema_payload(self, properties: dict) -> dict:
        out: dict[str, Any] = {}
        for name, spec in properties.items():
            kind = spec["type"]
            if kind in COMPUTED_PROPERTY_TYPES:
                self.notes.append(f"dropped computed property {name!r} ({kind})")
                continue
            if kind == "status":
                # `status` is not creatable through the API.
                options = strip_option_ids(spec["status"].get("options") or [])
                out[name] = {"select": {"options": options}}
                self.notes.append(f"property {name!r}: status -> select")
                continue
            body = strip_ids(spec.get(kind) or {})
            if kind in ("select", "multi_select") and body.get("options"):
                body["options"] = strip_option_ids(body["options"])
            out[name] = {kind: body}
        return out

    def row_payload(self, properties: dict, schema: dict) -> dict:
        out: dict[str, Any] = {}
        for name, value in properties.items():
            if name not in schema:
                continue
            kind = value["type"]
            target = next(iter(schema[name]))  # the type we actually created
            data = value.get(kind)
            if data in (None, [], {}):
                continue
            if kind == "status":
                out[name] = {"select": {"name": data["name"]}} if target == "select" \
                    else {"status": {"name": data["name"]}}
            elif kind in ("select",):
                out[name] = {"select": {"name": data["name"]}}
            elif kind == "multi_select":
                out[name] = {"multi_select": [{"name": o["name"]} for o in data]}
            elif kind == "people":
                out[name] = {"people": [{"object": "user", "id": p["id"]} for p in data]}
            elif kind == "files":
                external = [f for f in data if f.get("type") == "external"]
                if external:
                    out[name] = {"files": strip_ids(external)}
            else:
                out[name] = {kind: strip_ids(data)}
        return out

    def copy_database(self, src_db_id: str, dst_page_id: str) -> str:
        src = self.client.request("GET", f"/databases/{src_db_id}")
        schema = self.schema_payload(src["properties"])
        body = {
            "parent": {"type": "page_id", "page_id": dst_page_id},
            "title": strip_ids(src.get("title") or []),
            "properties": schema,
            "is_inline": src.get("is_inline", True),
        }
        if src.get("description"):
            body["description"] = strip_ids(src["description"])
        for key in ("icon", "cover"):
            decoration = externalize(src.get(key))
            if decoration:
                body[key] = decoration

        try:
            new_db = self.client.request("POST", "/databases", body)
        except NotionError as exc:
            # A database inside a column renders fine but cannot be created
            # there: "Parent block type column cannot contain databases".
            # Hoisting it to the enclosing page keeps the data and keeps it
            # reachable by the recursive child_database search the evaluators
            # use; only the two-column layout is lost.
            if "cannot contain databases" not in str(exc) or not self.page_id:
                raise
            self.notes.append("database hoisted out of a column onto its page "
                              "(the API cannot create one inside a column)")
            body["parent"] = {"type": "page_id", "page_id": self.page_id}
            new_db = self.client.request("POST", "/databases", body)
        for row in self.client.paged("POST", f"/databases/{src_db_id}/query"):
            new_row = self.client.request("POST", "/pages", {
                "parent": {"database_id": new_db["id"]},
                "properties": self.row_payload(row["properties"], schema),
            })
            # rows are pages; a template's rows sometimes carry body content
            if row.get("has_children") is not False:
                enclosing, self.page_id = self.page_id, new_row["id"]
                try:
                    self.copy_children(row["id"], new_row["id"])
                finally:
                    self.page_id = enclosing
        return new_db["id"]


# ---------- verification ----------


def fingerprint(client: Notion, page_id: str) -> dict:
    """A structural summary of a page, for comparing a copy to its template."""
    blocks: dict[str, int] = {}
    databases: list[dict] = []

    def walk(block_id: str, depth: int = 0) -> None:
        for block in client.children(block_id):
            kind = block["type"]
            blocks[kind] = blocks.get(kind, 0) + 1
            if kind == "child_database":
                db = client.request("GET", f"/databases/{block['id']}")
                rows = client.paged("POST", f"/databases/{block['id']}/query")
                databases.append({
                    "title": "".join(t["plain_text"] for t in db.get("title") or []),
                    "properties": {k: v["type"] for k, v in db["properties"].items()},
                    "rows": len(rows),
                    "titles": sorted(
                        "".join(t["plain_text"] for t in (p.get("title") or []))
                        for row in rows
                        for p in row["properties"].values() if p["type"] == "title"
                    ),
                })
            elif block.get("has_children") and depth < 8:
                walk(block["id"], depth + 1)

    walk(page_id)
    return {"blocks": blocks, "databases": sorted(databases, key=lambda d: d["title"])}


def compare(src: dict, dst: dict) -> list[str]:
    problems = []
    src_blocks = dict(src["blocks"])
    dst_blocks = dict(dst["blocks"])
    # the two documented substitutions
    if src_blocks.pop("link_preview", 0):
        dst_blocks["bookmark"] = dst_blocks.get("bookmark", 0) - src["blocks"]["link_preview"]
        if dst_blocks["bookmark"] <= 0:
            dst_blocks.pop("bookmark")
    for kind in set(src_blocks) | set(dst_blocks):
        if src_blocks.get(kind, 0) != dst_blocks.get(kind, 0):
            problems.append(
                f"block {kind}: template {src_blocks.get(kind, 0)} vs copy {dst_blocks.get(kind, 0)}"
            )
    if len(src["databases"]) != len(dst["databases"]):
        problems.append(f"database count {len(src['databases'])} vs {len(dst['databases'])}")
        return problems
    for a, b in zip(src["databases"], dst["databases"]):
        if a["title"] != b["title"]:
            problems.append(f"database title {a['title']!r} vs {b['title']!r}")
        if a["rows"] != b["rows"]:
            problems.append(f"{a['title']}: rows {a['rows']} vs {b['rows']}")
        if a["titles"] != b["titles"]:
            problems.append(f"{a['title']}: row titles differ")
        for name, kind in a["properties"].items():
            other = b["properties"].get(name)
            if other != kind and not (kind == "status" and other == "select"):
                problems.append(f"{a['title']}.{name}: {kind} vs {other}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-parent", required=True)
    ap.add_argument("--child-name", required=True)
    ap.add_argument("--target-parent", required=True)
    ap.add_argument("--notion-key", required=True)
    ap.add_argument("--output-file")
    ap.add_argument("--verify", action="store_true",
                    help="compare the copy against the template and fail on a mismatch")
    args = ap.parse_args()

    client = Notion(args.notion_key)
    source_parent = page_id_from_url(args.source_parent)
    target_parent = page_id_from_url(args.target_parent)

    template = next(
        (b for b in client.children(source_parent)
         if b["type"] == "child_page" and b["child_page"]["title"] == args.child_name),
        None,
    )
    if template is None:
        print(f"ERROR: no child page named {args.child_name!r} under the source parent")
        return 1
    print(f"template page: {template['id']}")

    notes: list[str] = []
    new_id = Duplicator(client, notes).copy_page(
        template["id"], target_parent, title=args.child_name
    )
    print(f"duplicated page id: {new_id}")
    for note in dict.fromkeys(notes):
        print(f"  note: {note}")

    if args.verify:
        problems = compare(fingerprint(client, template["id"]),
                           fingerprint(client, new_id))
        if problems:
            print("VERIFY FAILED:")
            for p in problems:
                print(f"  - {p}")
            return 2
        print("VERIFY OK: copy matches the template")

    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(new_id)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
