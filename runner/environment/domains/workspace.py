"""Workspace domain: file organization, documents, trackers.

move_files / archive_files take a folder_id (not a free-text
destination / archive_location). New ref type `folder` (shared contract with
communication.folder). Each session env that uses move/archive must preseed
a folder pool.

Natural enums `scheme` (classify_files) and `update_action` (update_tracker)
stay in ontology.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from runner.environment.base import (
    ATRDB, ATREnv, ATRToolKitBase, PersonaProfile, ToolType,
    _empty_with_hint, _loose_string_match, is_tool,
)
from runner.environment.tables import TableSpec, register_domain_tables
from runner.environment._validators import parse_enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class FileRecord(BaseModel):
    file_id: str
    name: str
    file_type: str
    location: str
    owner: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    archived: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    document_id: str
    title: str
    doc_type: str
    location: str
    content_brief: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TrackerEntry(BaseModel):
    entry_id: str
    status: str = "open"  # open / in_progress / closed
    next_action: Optional[str] = None
    payload: Optional[str] = None


class TrackerRecord(BaseModel):
    tracker_id: str
    title: str
    entries: dict[str, TrackerEntry] = Field(default_factory=dict)


class FolderRecord(BaseModel):
    """v0.12: destination folder for move_files / archive_files.
    Session env preseeds a folder pool; tool picks by id.
    """
    folder_id: str
    name: str
    purpose: Optional[str] = None


class WorkspaceDB(ATRDB):
    files: dict[str, FileRecord] = Field(default_factory=dict)
    documents: dict[str, DocumentRecord] = Field(default_factory=dict)
    trackers: dict[str, TrackerRecord] = Field(default_factory=dict)
    folders: dict[str, FolderRecord] = Field(default_factory=dict)

    _KNOWN_REF_TYPES = {"file", "document", "tracker", "folder"}

    _TABLES: list[TableSpec] = []

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

ClassifyScheme = Literal["by_project", "by_type", "by_date", "by_owner", "by_priority"]
TrackerUpdateAction = Literal["create_entry", "update_status", "record_next_action", "close_entry"]


class WorkspaceTools(ATRToolKitBase):
    db: WorkspaceDB

    def __init__(self, db: WorkspaceDB):
        super().__init__(db)

    # -------- Files --------

    @is_tool(ToolType.READ)
    def list_files(
        self,
        location: Optional[str] = None,
        file_type: Optional[Literal["pdf", "docx", "xlsx", "image", "txt", "csv"]] = None,
    ) -> dict:
        """List files or directory contents. Without `location`, returns the
        full session file pool; pass a substring to narrow.
        """
        pool = list(self.db.files.values())
        items = pool
        if location:
            items = [f for f in items if _loose_string_match(location, f.location)]
        if file_type:
            items = [f for f in items if _loose_string_match(file_type, f.file_type)]
        if not items and pool:
            return _empty_with_hint(
                f"location='{location}', file_type='{file_type}'",
                "location does substring/token match against file.location; "
                "try a parent path (e.g. '/' instead of '/Downloads'), or omit "
                "location/file_type to widen the search (omitting both returns "
                "the full file pool)",
            )
        results = [
            {"file_id": f.file_id, "name": f.name, "file_type": f.file_type,
             "location": f.location, "archived": f.archived,
             "owner": f.owner, "labels": f.labels,
             "attributes": f.attributes}
            for f in items
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.READ)
    def list_trackers(self) -> dict:
        """List session-scoped trackers. update_tracker.tracker_id must come from here."""
        return {
            "count": len(self.db.trackers),
            "results": [
                {"tracker_id": t.tracker_id, "title": t.title,
                 "entry_count": len(t.entries)}
                for t in self.db.trackers.values()
            ],
        }

    @is_tool(ToolType.READ)
    def list_file_folders(self) -> dict:
        """List session-scoped destination folders for file organization.
        move_files / archive_files.folder_id must come from here.
        """
        return {
            "count": len(self.db.folders),
            "results": [
                {"folder_id": f.folder_id, "name": f.name, "purpose": f.purpose}
                for f in self.db.folders.values()
            ],
        }

    @is_tool(ToolType.READ)
    def search_documents(
        self,
        location: Optional[str] = None,
        doc_type: Optional[Literal["note", "memo", "spreadsheet", "report", "template"]] = None,
    ) -> dict:
        """Search one-shot documents (notes, spreadsheets, reports). Does NOT
        include trackers — use list_trackers for ongoing logs / checklists.
        """
        pool = list(self.db.documents.values())
        items = pool
        if location:
            items = [d for d in items if _loose_string_match(location, d.location)]
        if doc_type:
            items = [d for d in items if _loose_string_match(doc_type, d.doc_type)]
        if not items and pool:
            return _empty_with_hint(
                f"location='{location}', doc_type='{doc_type}'",
                "filters do substring/token match against document.location and .doc_type; "
                "omit one filter, try a parent path, or different doc_type spelling",
            )
        return {
            "count": len(items),
            "results": [{"document_id": d.document_id, "title": d.title,
                         "doc_type": d.doc_type, "location": d.location,
                         "content_brief": d.content_brief,
                         "attributes": d.attributes}
                        for d in items],
        }

    @is_tool(ToolType.WRITE)
    def classify_files(
        self, file_ids: list[str], scheme: ClassifyScheme,
    ) -> dict:
        """Assign labels or categories to a set of files using a fixed scheme."""
        parse_enum(scheme, {"by_project", "by_type", "by_date", "by_owner", "by_priority"}, "scheme")
        _validate_ids(file_ids, self.db.files, "File")
        for fid in file_ids:
            f = self.db.files[fid]
            tag = f"{scheme}:{_scheme_value(f, scheme)}"
            if tag not in f.labels:
                f.labels.append(tag)
        return {"classified": len(file_ids), "scheme": scheme}

    @is_tool(ToolType.WRITE)
    def move_files(self, file_ids: list[str], folder_id: str) -> dict:
        """Move one or more files to a destination folder (by id).
        Session env preseeds a folder pool; agent picks one per rule/need.
        """
        folder = self.db.folders.get(folder_id)
        if not folder:
            raise ValueError(f"Folder not found: {folder_id}")
        _validate_ids(file_ids, self.db.files, "File")
        for fid in file_ids:
            self.db.files[fid].location = folder.name
        return {"moved": len(file_ids), "folder_id": folder_id, "name": folder.name}

    @is_tool(ToolType.WRITE)
    def archive_files(
        self, file_ids: list[str], folder_id: str,
    ) -> dict:
        """Archive one or more files into a destination folder (by id)."""
        folder = self.db.folders.get(folder_id)
        if not folder:
            raise ValueError(f"Folder not found: {folder_id}")
        _validate_ids(file_ids, self.db.files, "File")
        for fid in file_ids:
            f = self.db.files[fid]
            f.archived = True
            f.location = folder.name
        return {"archived": len(file_ids), "folder_id": folder_id, "name": folder.name}

    @is_tool(ToolType.WRITE)
    def delete_files(self, file_ids: list[str]) -> dict:
        """Delete one or more files."""
        _validate_ids(file_ids, self.db.files, "File")
        for fid in file_ids:
            del self.db.files[fid]
        return {"deleted": len(file_ids)}

    # -------- Documents --------

    @is_tool(ToolType.WRITE)
    def create_document(
        self,
        doc_type: Literal["note", "memo", "spreadsheet", "report", "template"],
        title: str, location: str,
        content_brief: Optional[str] = None,
    ) -> dict:
        """Create a document, spreadsheet, note, or template."""
        parse_enum(doc_type, {"note", "memo", "spreadsheet", "report", "template"}, "doc_type")
        seq = len(self.db.documents) + 1
        document_id = f"DOC_{seq:03d}"
        self.db.documents[document_id] = DocumentRecord(
            document_id=document_id, title=title, doc_type=doc_type,
            location=location, content_brief=content_brief,
        )
        return {"document_id": document_id, "status": "created"}

    @is_tool(ToolType.WRITE)
    def update_document(
        self, document_id: str,
        content_delta: str,
    ) -> dict:
        """Update an existing document, spreadsheet, or note. Appends
        content_delta to the document's content_brief.
        """
        d = self.db.documents.get(document_id)
        if not d:
            raise ValueError(f"Document not found: {document_id}")
        if d.content_brief is None:
            d.content_brief = content_delta
        else:
            d.content_brief = (d.content_brief or "") + "\n" + content_delta
        return {"document_id": document_id, "status": "updated"}

    # -------- Trackers --------

    @is_tool(ToolType.WRITE)
    def update_tracker(
        self,
        tracker_id: str,
        update_action: TrackerUpdateAction,
        entry_id: Optional[str] = None,
        entry_payload: Optional[str] = None,
    ) -> dict:
        """Maintain a structured tracker, log, or checklist."""
        parse_enum(update_action, {"create_entry", "update_status", "record_next_action", "close_entry"}, "update_action")
        t = self.db.trackers.get(tracker_id)
        if not t:
            raise ValueError(f"Tracker not found: {tracker_id}")
        if update_action == "create_entry":
            seq = len(t.entries) + 1
            eid = entry_id or f"ENT_{seq:03d}"
            t.entries[eid] = TrackerEntry(entry_id=eid, payload=entry_payload)
            return {"tracker_id": tracker_id, "entry_id": eid, "action": "created"}
        if entry_id is None:
            raise ValueError(f"entry_id required for action={update_action}")
        ent = t.entries.get(entry_id)
        if not ent:
            raise ValueError(f"Entry not found: {entry_id}")
        if update_action == "update_status":
            ent.status = "in_progress"
        elif update_action == "record_next_action":
            ent.next_action = entry_payload
        elif update_action == "close_entry":
            ent.status = "closed"
        return {"tracker_id": tracker_id, "entry_id": entry_id, "action": update_action}


def _validate_ids(ids: list[str], pool: dict, kind: str) -> None:
    """Raise on the first unknown id — enforces id_discovery_convention for
    multi-id write tools. Call before mutating so nothing is partially applied.
    """
    for i in ids:
        if i not in pool:
            raise ValueError(f"{kind} not found: {i}")


def _scheme_value(f: FileRecord, scheme: ClassifyScheme) -> str:
    if scheme == "by_project":
        return f.attributes.get("project", "unassigned")
    if scheme == "by_type":
        return f.file_type
    if scheme == "by_date":
        return f.attributes.get("modified_date", "unknown")
    if scheme == "by_owner":
        return f.owner or "unknown"
    if scheme == "by_priority":
        return f.attributes.get("priority", "normal")
    return "unknown"


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(
    session_task, persona: PersonaProfile, allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    db = WorkspaceDB.from_references(persona, session_task.local_env.references)
    toolkit = WorkspaceTools(db)
    return ATREnv(
        domain="workspace", toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------

_FILE_PROMOTED = frozenset({
    "name", "file_type", "location", "owner", "labels", "archived",
})


def _build_file_row(ref_id, attrs, persona, spec):
    return {
        "file_id": ref_id,
        "name": attrs.get("name", ref_id),
        "file_type": attrs.get("file_type", "other"),
        "location": attrs.get("location", "/"),
        "owner": attrs.get("owner"),
        "labels": attrs.get("labels", []),
        "archived": attrs.get("archived", False),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _FILE_PROMOTED},
    }


_DOC_PROMOTED = frozenset({"title", "doc_type", "location", "content_brief"})


def _build_document_row(ref_id, attrs, persona, spec):
    return {
        "document_id": ref_id,
        "title": attrs.get("title", ref_id),
        "doc_type": attrs.get("doc_type", "note"),
        "location": attrs.get("location", "/"),
        "content_brief": attrs.get("content_brief"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _DOC_PROMOTED},
    }


def _build_tracker_row(ref_id, attrs, persona, spec):
    raw_entries = attrs.get("entries", [])
    entries: dict[str, TrackerEntry] = {}
    if isinstance(raw_entries, list):
        for e in raw_entries:
            if not isinstance(e, dict) or "entry_id" not in e:
                continue
            eid = e["entry_id"]
            entries[eid] = TrackerEntry(
                entry_id=eid,
                status=e.get("status", "open"),
                next_action=e.get("next_action"),
                payload=e.get("payload"),
            )
    return {
        "tracker_id": ref_id,
        "title": attrs.get("title", ref_id),
        "entries": entries,
    }


def _build_folder_row(ref_id, attrs, persona, spec):
    return {
        "folder_id": ref_id,
        "name": attrs.get("name", ref_id),
        "purpose": attrs.get("purpose"),
    }


_WORKSPACE_TABLES = [
    TableSpec(
        name="files", model=FileRecord, kind="primary",
        source_ref_type="file",
        promoted_attrs=["name", "file_type", "location", "owner", "labels", "archived"],
        operating_tools=["list_files", "classify_files",
                         "move_files", "archive_files", "delete_files"],
        discovery_tools=["list_files"],
        build_row=_build_file_row,
    ),
    TableSpec(
        name="documents", model=DocumentRecord, kind="primary",
        source_ref_type="document",
        promoted_attrs=["title", "doc_type", "location", "content_brief"],
        operating_tools=["search_documents", "update_document", "create_document"],
        discovery_tools=["search_documents"],
        build_row=_build_document_row,
    ),
    TableSpec(
        name="trackers", model=TrackerRecord, kind="primary",
        source_ref_type="tracker",
        promoted_attrs=["title"],
        operating_tools=["list_trackers", "update_tracker"],
        discovery_tools=["list_trackers"],
        build_row=_build_tracker_row,
    ),
    TableSpec(
        name="folders", model=FolderRecord, kind="primary",
        source_ref_type="folder",
        promoted_attrs=["name", "purpose"],
        operating_tools=["list_file_folders", "move_files", "archive_files"],
        discovery_tools=["list_file_folders"],
        build_row=_build_folder_row,
    ),
]

WorkspaceDB._TABLES = _WORKSPACE_TABLES
register_domain_tables("workspace", _WORKSPACE_TABLES)
