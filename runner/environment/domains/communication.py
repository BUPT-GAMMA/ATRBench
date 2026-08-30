"""Communication domain: email / message search, prioritization, drafting, sending.

Ref types: message, folder, label. Each session env preseeds folder / label
pools as needed in local_env.references; messages are also lifted from refs
of type 'message'. set_message_priority takes a flat enum value
(high/medium/low/normal) rather than a bucket id.
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

class MessageRecord(BaseModel):
    message_id: str
    sender: str
    subject: str
    body: str = ""
    thread_id: Optional[str] = None
    # System folder enum: inbox / archive / sent / drafts. Distinct from
    # the user-defined folder pool (FolderRecord) which lives on
    # `assigned_folder_id`.
    folder: str = "inbox"
    labels: list[str] = Field(default_factory=list)
    priority: str = "normal"  # "high" / "medium" / "normal" / "low"
    archived: bool = False
    # Optional user-defined folder id assigned by archive_messages.
    # Kept separate from `folder` so the system-folder enum stays clean
    # for search_messages.folder lookups.
    assigned_folder_id: Optional[str] = None
    timestamp: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class FolderRecord(BaseModel):
    folder_id: str
    name: str
    purpose: Optional[str] = None


class LabelRecord(BaseModel):
    label_id: str
    name: str
    topic: Optional[str] = None


class DraftRecord(BaseModel):
    draft_id: str
    recipient: Optional[str] = None
    subject: Optional[str] = None
    body: str
    thread_id: Optional[str] = None
    language: Optional[str] = None


class SentMessage(BaseModel):
    sent_id: str
    recipient: Optional[str] = None
    subject: Optional[str] = None
    body: str
    thread_id: Optional[str] = None
    language: Optional[str] = None
    status: str = "sent"


class CommunicationDB(ATRDB):
    messages: dict[str, MessageRecord] = Field(default_factory=dict)
    folders: dict[str, FolderRecord] = Field(default_factory=dict)
    labels: dict[str, LabelRecord] = Field(default_factory=dict)
    drafts: dict[str, DraftRecord] = Field(default_factory=dict)
    sent: dict[str, SentMessage] = Field(default_factory=dict)

    _KNOWN_REF_TYPES = {"message", "folder", "label"}

    _TABLES: list[TableSpec] = []

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

PriorityLevel = Literal["high", "medium", "low", "normal"]


class CommunicationTools(ATRToolKitBase):
    db: CommunicationDB
    _shape_guards = {
        "send_message": {"at_least_one_of": ["recipient", "thread_id"]},
        "draft_message": {"at_least_one_of": ["recipient", "thread_id"]},
    }

    def __init__(self, db: CommunicationDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def search_messages(
        self,
        sender: Optional[str] = None,
        folder: Optional[Literal["inbox", "archive", "sent", "drafts"]] = None,
    ) -> dict:
        """Search email or message threads. The `folder` filter targets email
        system folders (inbox/archive/sent/drafts), distinct from the
        user-defined folders listed by `list_message_folders` (whose
        folder_id slot is consumed by archive_messages). Inspect each
        returned message's `timestamp` to pick which one matches a time
        description in the instruction.

        When `folder="archive"`, archived messages are returned (they live
        in the archive system folder). For every other folder lookup,
        archived messages stay hidden so the active inbox view is clean.
        """
        pool = list(self.db.messages.values())
        items = pool
        if sender:
            items = [m for m in items if _loose_string_match(sender, m.sender)]
        if folder:
            if folder == "archive":
                items = [m for m in items if m.archived]
            else:
                items = [m for m in items if _loose_string_match(folder, m.folder)]
                items = [m for m in items if not m.archived]
        else:
            items = [m for m in items if not m.archived]
        if not items and pool:
            return _empty_with_hint(
                f"sender='{sender}', folder='{folder}'",
                "filters do substring/token match against message.sender and .folder; "
                "try omitting one filter to see the inbox",
            )
        results = [
            {"message_id": m.message_id, "sender": m.sender, "subject": m.subject,
             "folder": m.folder, "priority": m.priority, "thread_id": m.thread_id,
             "labels": m.labels, "timestamp": m.timestamp,
             "archived": m.archived,
             "assigned_folder_id": m.assigned_folder_id,
             "attributes": m.attributes}
            for m in items
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.READ)
    def list_labels(self) -> dict:
        """List session-scoped labels. label_messages.label_ids must come from here."""
        return {
            "count": len(self.db.labels),
            "results": [
                {"label_id": l.label_id, "name": l.name, "topic": l.topic}
                for l in self.db.labels.values()
            ],
        }

    @is_tool(ToolType.READ)
    def list_message_folders(self) -> dict:
        """List session-scoped email folders (user-defined). Distinct from
        the email system folders (inbox/archive/sent/drafts) targeted by
        search_messages.folder. archive_messages.folder_id must come from
        here.
        """
        return {
            "count": len(self.db.folders),
            "results": [
                {"folder_id": f.folder_id, "name": f.name, "purpose": f.purpose}
                for f in self.db.folders.values()
            ],
        }

    @is_tool(ToolType.WRITE)
    def set_message_priority(
        self, message_ids: list[str], priority: PriorityLevel,
    ) -> dict:
        """Assign a priority level (high / medium / low / normal) to one or
        more messages.
        """
        parse_enum(priority, {"high", "medium", "low", "normal"}, "priority")
        _validate_ids(message_ids, self.db.messages, "Message")
        for mid in message_ids:
            self.db.messages[mid].priority = priority
        return {
            "prioritized": len(message_ids),
            "priority": priority,
        }

    @is_tool(ToolType.WRITE)
    def archive_messages(
        self, message_ids: list[str], folder_id: Optional[str] = None,
    ) -> dict:
        """Archive one or more messages out of the active queue.
        Optionally pin them to a specific user-defined folder by id.

        The system-folder enum on each message (inbox / archive / sent /
        drafts) is set to "archive" so subsequent
        `search_messages(folder="archive")` can find them. The
        user-defined folder selection lives separately on
        `assigned_folder_id` and does not clobber the system-folder slot.
        """
        if folder_id is not None:
            folder = self.db.folders.get(folder_id)
            if not folder:
                raise ValueError(f"Folder not found: {folder_id}")
        _validate_ids(message_ids, self.db.messages, "Message")
        for mid in message_ids:
            m = self.db.messages[mid]
            m.archived = True
            m.folder = "archive"
            if folder_id is not None:
                m.assigned_folder_id = folder_id
        return {"archived": len(message_ids), "folder_id": folder_id}

    def _validate_thread_id(self, thread_id: Optional[str]) -> None:
        """Enforce id_discovery_convention for thread_id: if the agent passes
        one, it must belong to an existing message's thread. Agents discover
        thread_ids by calling search_messages first.
        """
        if thread_id is None:
            return
        valid = {m.thread_id for m in self.db.messages.values() if m.thread_id}
        if thread_id not in valid:
            raise ValueError(f"Thread not found: {thread_id}")

    @is_tool(ToolType.WRITE)
    def draft_message(
        self,
        body: str,
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
        thread_id: Optional[str] = None,
        language: Optional[Literal["en", "zh"]] = None,
    ) -> dict:
        """Draft a new message or a reply to an existing thread.
        Shape guard (env-enforced) — at least one of recipient / thread_id
        must be provided. If thread_id is passed it must reference an
        existing message's thread (discover via search_messages).
        """
        self._validate_thread_id(thread_id)
        if language is not None:
            parse_enum(language, {"en", "zh"}, "language")
        seq = len(self.db.drafts) + 1
        draft_id = f"DRAFT_{seq:03d}"
        self.db.drafts[draft_id] = DraftRecord(
            draft_id=draft_id, recipient=recipient, subject=subject,
            body=body, thread_id=thread_id, language=language,
        )
        return {"draft_id": draft_id, "status": "drafted"}

    @is_tool(ToolType.WRITE)
    def send_message(
        self,
        body: str,
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
        thread_id: Optional[str] = None,
        language: Optional[Literal["en", "zh"]] = None,
    ) -> dict:
        """Send a new message or a reply to an existing thread.
        Shape guard (env-enforced) — at least one of recipient / thread_id
        must be provided (no address-less sends). If thread_id is passed
        it must reference an existing message's thread.
        """
        self._validate_thread_id(thread_id)
        if language is not None:
            parse_enum(language, {"en", "zh"}, "language")
        seq = len(self.db.sent) + 1
        sent_id = f"SENT_{seq:03d}"
        self.db.sent[sent_id] = SentMessage(
            sent_id=sent_id, recipient=recipient, subject=subject,
            body=body, thread_id=thread_id, language=language,
        )
        return {"sent_id": sent_id, "status": "sent"}

    @is_tool(ToolType.WRITE)
    def label_messages(
        self, message_ids: list[str], label_ids: list[str],
    ) -> dict:
        """Apply one or more labels (identified by id) to message threads.
        Each session preseeds a label pool in references.
        """
        _validate_ids(label_ids, self.db.labels, "Label")
        _validate_ids(message_ids, self.db.messages, "Message")
        resolved = [self.db.labels[lid].name for lid in label_ids]
        for mid in message_ids:
            m = self.db.messages[mid]
            m.labels = list(dict.fromkeys(m.labels + resolved))
        return {"labeled": len(message_ids), "label_ids": label_ids, "names": resolved}

    @is_tool(ToolType.WRITE)
    def send_draft(
        self, draft_id: str,
    ) -> dict:
        """Send a previously created draft message. Use this (not send_message)
        when the workflow is draft → user review → send.
        """
        d = self.db.drafts.get(draft_id)
        if not d:
            raise ValueError(f"Draft not found: {draft_id}")
        seq = len(self.db.sent) + 1
        sent_id = f"SENT_{seq:03d}"
        self.db.sent[sent_id] = SentMessage(
            sent_id=sent_id, recipient=d.recipient, subject=d.subject,
            body=d.body, thread_id=d.thread_id, language=d.language,
        )
        return {
            "sent_id": sent_id, "status": "sent",
            "from_draft_id": draft_id,
        }


def _validate_ids(ids: list[str], pool: dict, kind: str) -> None:
    """Raise on the first unknown id — enforces id_discovery_convention for
    multi-id write tools. Call before mutating so nothing is partially applied.
    """
    for i in ids:
        if i not in pool:
            raise ValueError(f"{kind} not found: {i}")


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(
    session_task, persona: PersonaProfile, allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    db = CommunicationDB.from_references(persona, session_task.local_env.references)
    toolkit = CommunicationTools(db)
    return ATREnv(
        domain="communication", toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------

_MESSAGE_PROMOTED = frozenset({
    "sender", "subject", "body", "thread_id",
    "folder", "labels", "priority", "archived", "timestamp",
})


def _build_message_row(ref_id, attrs, persona, spec):
    return {
        "message_id": ref_id,
        "sender": attrs.get("sender", "unknown"),
        "subject": attrs.get("subject", ""),
        "body": attrs.get("body", ""),
        "thread_id": attrs.get("thread_id"),
        "folder": attrs.get("folder", "inbox"),
        "labels": attrs.get("labels", []),
        "priority": attrs.get("priority", "normal"),
        "archived": attrs.get("archived", False),
        "timestamp": attrs.get("timestamp"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _MESSAGE_PROMOTED},
    }


def _build_folder_row(ref_id, attrs, persona, spec):
    return {
        "folder_id": ref_id,
        "name": attrs.get("name", ref_id),
        "purpose": attrs.get("purpose"),
    }


def _build_label_row(ref_id, attrs, persona, spec):
    return {
        "label_id": ref_id,
        "name": attrs.get("name", ref_id),
        "topic": attrs.get("topic"),
    }


_COMMUNICATION_TABLES = [
    TableSpec(
        name="messages", model=MessageRecord, kind="primary",
        source_ref_type="message",
        promoted_attrs=["sender", "subject", "body", "thread_id",
                        "folder", "labels", "priority", "archived", "timestamp"],
        operating_tools=["search_messages", "set_message_priority",
                         "archive_messages", "label_messages",
                         "draft_message", "send_message", "send_draft"],
        discovery_tools=["search_messages"],
        build_row=_build_message_row,
    ),
    TableSpec(
        name="folders", model=FolderRecord, kind="primary",
        source_ref_type="folder",
        promoted_attrs=["name", "purpose"],
        operating_tools=["list_message_folders", "archive_messages"],
        discovery_tools=["list_message_folders"],
        build_row=_build_folder_row,
    ),
    TableSpec(
        name="labels", model=LabelRecord, kind="primary",
        source_ref_type="label",
        promoted_attrs=["name", "topic"],
        operating_tools=["list_labels", "label_messages"],
        discovery_tools=["list_labels"],
        build_row=_build_label_row,
    ),
]

CommunicationDB._TABLES = _COMMUNICATION_TABLES
register_domain_tables("communication", _COMMUNICATION_TABLES)
