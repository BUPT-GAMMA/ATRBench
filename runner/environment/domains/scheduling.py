"""Scheduling domain: calendar events, conflicts, reminders, coordination.

Natural enums (`response`, `cadence`, `update_scope`) stay in ontology.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from runner.environment.base import (
    ATRDB, ATREnv, ATRToolKitBase, PersonaProfile, ToolType,
    _empty_with_hint, _loose_string_match, is_tool,
)
from runner.environment.tables import TableSpec, register_domain_tables
from runner.environment._validators import parse_iso_datetime, parse_enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class CalendarEvent(BaseModel):
    event_id: str
    title: str
    start_time: str
    end_time: str
    participants: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    calendar: str = "default"
    status: str = "confirmed"  # confirmed / declined / tentative / cancelled
    attributes: dict[str, Any] = Field(default_factory=dict)


class Reminder(BaseModel):
    reminder_id: str
    target_id: str
    remind_at: Optional[str] = None
    cadence: Optional[str] = None


class SchedulingDB(ATRDB):
    events: dict[str, CalendarEvent] = Field(default_factory=dict)
    reminders: dict[str, Reminder] = Field(default_factory=dict)

    # "event" alias is accepted for call-site brevity; "calendar_event" is canonical.
    _KNOWN_REF_TYPES = {"calendar_event", "event"}

    _TABLES: list[TableSpec] = []

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

InviteResponse = Literal["accept", "decline", "tentative"]
Cadence = Literal["once", "daily", "weekly", "monthly", "before_event"]
UpdateScope = Literal["critical", "all", "schedule_change"]


class SchedulingTools(ATRToolKitBase):
    db: SchedulingDB

    # Fields modify_event is allowed to write. Time changes route through
    # reschedule_event, status through cancel_event / respond_to_event_invite.
    _MODIFIABLE_FIELDS: dict[str, set[str]] = {
        "modify_event": {"title", "location", "participants"},
    }

    def __init__(self, db: SchedulingDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def list_events(
        self,
        calendar: Optional[Literal["primary", "work", "personal", "invites"]] = None,
        participants: Optional[list[str]] = None,
    ) -> dict:
        """List calendar events. Returns the full session calendar; inspect
        each returned event's `start_time` / `end_time` to pick which one
        matches the user's instruction time description.

        Filters:
          - calendar: substring/token match against event.calendar
          - participants: must be a list of EMAIL addresses (the canonical
            participant id) — names like "Nina" will not match. Prefer
            omitting and post-filtering on the returned list unless you
            have the exact email.
        """
        pool = list(self.db.events.values())
        items = pool
        if calendar:
            items = [e for e in items if _loose_string_match(calendar, e.calendar)]
        if participants:
            pset = set(p.lower() for p in participants)
            items = [e for e in items
                     if any(p.lower() in pset for p in e.participants)]
        active = [e for e in items if e.status != "cancelled"]
        if not active and pool:
            return _empty_with_hint(
                f"calendar='{calendar}', participants={participants}",
                "calendar / participants do substring/email match; try omitting one filter",
            )
        results = [
            {"event_id": e.event_id, "title": e.title,
             "start_time": e.start_time, "end_time": e.end_time,
             "participants": e.participants, "location": e.location,
             "status": e.status, "calendar": e.calendar,
             "attributes": e.attributes}
            for e in active
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.READ)
    def check_conflicts(
        self, start_time: str, end_time: str,
        participants: Optional[list[str]] = None,
    ) -> dict:
        """Check whether a proposed schedule conflicts with existing events.

        Minimal string-compare on timestamps; assumes ISO 8601.
        """
        start_time = parse_iso_datetime(start_time, "start_time")
        end_time = parse_iso_datetime(end_time, "end_time")
        conflicts = []
        for e in self.db.events.values():
            if e.status == "cancelled":
                continue
            if e.start_time < end_time and e.end_time > start_time:
                if participants is None or any(p in e.participants for p in participants):
                    conflicts.append({
                        "event_id": e.event_id, "title": e.title,
                        "start_time": e.start_time, "end_time": e.end_time,
                    })
        return {"conflicts": conflicts, "has_conflict": len(conflicts) > 0}

    @is_tool(ToolType.WRITE)
    def create_event(
        self, title: str, start_time: str, end_time: str,
        participants: Optional[list[str]] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Create a new calendar event."""
        start_time = parse_iso_datetime(start_time, "start_time")
        end_time = parse_iso_datetime(end_time, "end_time")
        if end_time <= start_time:
            raise ValueError(
                f"end_time ({end_time}) must be strictly after "
                f"start_time ({start_time})."
            )
        seq = len(self.db.events) + 1
        event_id = f"EVT_{seq:03d}"
        self.db.events[event_id] = CalendarEvent(
            event_id=event_id, title=title, start_time=start_time,
            end_time=end_time, participants=participants or [], location=location,
        )
        return {"event_id": event_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def reschedule_event(
        self, event_id: str, new_start_time: str, new_end_time: str,
    ) -> dict:
        """Reschedule an existing calendar event (time-only change)."""
        new_start_time = parse_iso_datetime(new_start_time, "new_start_time")
        new_end_time = parse_iso_datetime(new_end_time, "new_end_time")
        if new_end_time <= new_start_time:
            raise ValueError(
                f"new_end_time ({new_end_time}) must be strictly after "
                f"new_start_time ({new_start_time})."
            )
        e = self.db.events.get(event_id)
        if not e:
            raise ValueError(f"Event not found: {event_id}")
        e.start_time = new_start_time
        e.end_time = new_end_time
        return {"event_id": event_id, "status": "rescheduled"}

    @is_tool(ToolType.WRITE)
    def modify_event(
        self, event_id: str,
        field: Literal["title", "location", "participants"],
        new_value: str,
    ) -> dict:
        """Modify a calendar event's non-time fields (title, location,
        participants). Use reschedule_event for time changes, cancel_event
        for status. Only fields in _MODIFIABLE_FIELDS["modify_event"] are
        accepted.
        """
        allowed = self._MODIFIABLE_FIELDS.get("modify_event", set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable on an event. "
                f"Allowed: {sorted(allowed)}. "
                f"For time changes use reschedule_event; for status use "
                f"cancel_event or respond_to_event_invite."
            )
        e = self.db.events.get(event_id)
        if not e:
            raise ValueError(f"Event not found: {event_id}")
        if field == "participants":
            e.participants = [p.strip() for p in new_value.split(",") if p.strip()]
        else:
            setattr(e, field, new_value)
        return {"event_id": event_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def respond_to_event_invite(
        self, event_id: str, response: InviteResponse,
    ) -> dict:
        """Accept, decline, or tentatively respond to an event invitation."""
        parse_enum(response, {"accept", "decline", "tentative"}, "response")
        e = self.db.events.get(event_id)
        if not e:
            raise ValueError(f"Event not found: {event_id}")
        e.status = {"accept": "confirmed", "decline": "declined",
                    "tentative": "tentative"}[response]
        return {"event_id": event_id, "response": response}

    @is_tool(ToolType.WRITE)
    def cancel_event(self, event_id: str) -> dict:
        """Cancel an existing calendar event."""
        e = self.db.events.get(event_id)
        if not e:
            raise ValueError(f"Event not found: {event_id}")
        e.status = "cancelled"
        return {"event_id": event_id, "status": "cancelled"}

    @is_tool(ToolType.WRITE)
    def set_reminder(
        self, target_id: str,
        remind_at: Optional[str] = None,
        cadence: Optional[Cadence] = None,
    ) -> dict:
        """Create or update a reminder for an event, task, bill, or recurring
        obligation. target_id must reference an event surfaced by a prior
        list_events / create_event / reschedule_event call in this session.
        """
        if target_id not in self.db.events:
            raise ValueError(f"Event not found: {target_id}")
        if remind_at is not None:
            remind_at = parse_iso_datetime(remind_at, "remind_at")
        if cadence is not None:
            parse_enum(cadence, {"once", "daily", "weekly", "monthly", "before_event"}, "cadence")
        seq = len(self.db.reminders) + 1
        reminder_id = f"RMD_{seq:03d}"
        self.db.reminders[reminder_id] = Reminder(
            reminder_id=reminder_id, target_id=target_id,
            remind_at=remind_at, cadence=cadence,
        )
        return {"reminder_id": reminder_id, "target_id": target_id,
                "cadence": cadence, "remind_at": remind_at}

    @is_tool(ToolType.READ)
    def track_event_updates(
        self, target_id: str,
        update_scope: UpdateScope = "critical",
    ) -> dict:
        """Track changes to an event or activity and surface important updates.
        target_id must reference an event that exists in the session (from
        list_events / create_event).
        """
        parse_enum(update_scope, {"critical", "all", "schedule_change"}, "update_scope")
        if target_id not in self.db.events:
            raise ValueError(f"Event not found: {target_id}")
        return {
            "tracking_id": f"TRK_{target_id}",
            "target": target_id, "scope": update_scope,
            "status": "tracking_enabled",
        }

    @is_tool(ToolType.READ)
    def find_alternative_slots(self, target_id: str) -> dict:
        """Find nearby alternative times or substitute instances for an
        affected activity. target_id must reference an existing event.
        """
        target = self.db.events.get(target_id)
        if target is None:
            raise ValueError(f"Event not found: {target_id}")
        alternatives = [
            {"slot_id": f"ALT_{target_id}_{i}", "start_hint": f"{target.start_time}+{i}d"}
            for i in (1, 2, 3)
        ]
        return {"target_id": target_id, "alternatives": alternatives}



# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(
    session_task, persona: PersonaProfile, allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    db = SchedulingDB.from_references(persona, session_task.local_env.references)
    toolkit = SchedulingTools(db)
    return ATREnv(
        domain="scheduling", toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------

_EVENT_PROMOTED = frozenset({
    "title", "name", "start_time", "end_time",
    "participants", "location", "calendar", "status",
})


def _build_event_row(ref_id, attrs, persona, spec):
    return {
        "event_id": ref_id,
        "title": attrs.get("title") or attrs.get("name", ref_id),
        "start_time": attrs.get("start_time", ""),
        "end_time": attrs.get("end_time", ""),
        "participants": attrs.get("participants", []),
        "location": attrs.get("location"),
        "calendar": attrs.get("calendar", "default"),
        "status": attrs.get("status", "confirmed"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _EVENT_PROMOTED},
    }


_SCHEDULING_TABLES = [
    TableSpec(
        name="events", model=CalendarEvent, kind="primary",
        source_ref_type="calendar_event",
        aliases=["event"],
        promoted_attrs=["title", "start_time", "end_time",
                        "participants", "location", "calendar", "status"],
        operating_tools=[
            "list_events", "check_conflicts", "create_event",
            "reschedule_event", "modify_event",
            "respond_to_event_invite", "cancel_event",
            "set_reminder", "track_event_updates",
            "find_alternative_slots",
        ],
        discovery_tools=["list_events", "check_conflicts",
                         "find_alternative_slots"],
        build_row=_build_event_row,
    ),
]

SchedulingDB._TABLES = _SCHEDULING_TABLES
register_domain_tables("scheduling", _SCHEDULING_TABLES)
