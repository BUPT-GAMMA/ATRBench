"""Reservation domain: restaurants, events, local services.

Tools (yaml-aligned):
  search_restaurants / book_restaurant / modify_restaurant_reservation / cancel_restaurant_reservation
  search_events / book_event_ticket / modify_event_ticket / cancel_event_ticket
  search_service_providers / book_service_appointment / modify_service_appointment / cancel_service_appointment

Write tools auto-fill identity fields (contact_info) from persona.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from runner.environment.base import (
    ATRDB,
    ATREnv,
    ATRToolKitBase,
    PersonaProfile,
    ToolType,
    _empty_with_hint,
    _loose_string_match,
    is_tool,
)
from runner.environment.tables import TableSpec, register_domain_tables
from runner.environment._validators import parse_iso_datetime, parse_int, parse_enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Restaurant(BaseModel):
    restaurant_id: str
    name: str
    cuisine: str
    location: str
    rating: Optional[float] = None
    price_range: Optional[str] = None
    # v0.12: neutral-fact seating environment descriptor. Rules about seating
    # ambience land on restaurant_id selection via this attribute, not on a
    # book_restaurant param.
    seating_style: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
    event_id: str
    name: str
    location: str
    date_time: Optional[str] = None
    price: Optional[float] = None
    event_tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ServiceProvider(BaseModel):
    provider_id: str
    name: str
    service_type: str
    location: str
    rating: Optional[float] = None
    distance_km: Optional[float] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class RestaurantReservation(BaseModel):
    reservation_id: str
    restaurant_id: str
    date_time: str
    party_size: int
    contact_info: str
    status: str = "confirmed"


class EventBooking(BaseModel):
    booking_id: str
    event_id: str
    ticket_type: str
    ticket_count: int
    contact_info: str
    status: str = "confirmed"


class ServiceAppointment(BaseModel):
    appointment_id: str
    provider_id: str
    service_type: str
    date_time: str
    contact_info: str
    status: str = "confirmed"


class ReservationDB(ATRDB):
    restaurants: dict[str, Restaurant] = Field(default_factory=dict)
    events: dict[str, EventRecord] = Field(default_factory=dict)
    providers: dict[str, ServiceProvider] = Field(default_factory=dict)
    restaurant_reservations: dict[str, RestaurantReservation] = Field(default_factory=dict)
    event_bookings: dict[str, EventBooking] = Field(default_factory=dict)
    service_appointments: dict[str, ServiceAppointment] = Field(default_factory=dict)

    _KNOWN_REF_TYPES = {"restaurant", "event", "service_provider"}

    _TABLES: list[TableSpec] = []

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

ServiceSortBy = Literal["relevance", "distance", "rating"]
RestaurantSortBy = Literal["relevance", "rating", "price_asc", "price_desc"]


TrackingScope = Literal["critical", "all", "schedule_change"]


class ReservationTools(ATRToolKitBase):
    db: ReservationDB
    _shape_guards = {
        "track_reservation_updates": {
            "exactly_one_of": ["reservation_id", "booking_id", "appointment_id"],
        },
    }

    # Fields modify_* tools are allowed to write, keyed by tool name.
    # Any field outside this set is rejected at runtime — the generic
    # (field,new_value) signature can otherwise bypass cancel_* / status.
    _MODIFIABLE_FIELDS: dict[str, set[str]] = {
        "modify_restaurant_reservation": {"date_time", "party_size", "contact_info"},
        "modify_event_ticket": {"ticket_type", "ticket_count", "contact_info"},
        "modify_service_appointment": {"date_time", "contact_info"},
    }

    def __init__(self, db: ReservationDB):
        super().__init__(db)

    # -------- Restaurants --------

    @is_tool(ToolType.READ)
    def search_restaurants(
        self,
        location: str,
        cuisine: Optional[str] = None,
        sort_by: RestaurantSortBy = "relevance",
    ) -> dict:
        """Search restaurants that can be reserved. Inspect each returned
        restaurant's available slots / capacity to pick one for the user's
        date_time and party_size at booking time.

        Args:
            location: City or area to search in.
            cuisine: Optional cuisine filter.
            sort_by: How to rank results. Default 'relevance' preserves the
                     catalog's natural order; 'rating' / 'price_asc' /
                     'price_desc' give explicit orderings.
        """
        pool = list(self.db.restaurants.values())
        items = pool
        if location:
            items = [r for r in items if _loose_string_match(location, r.location)]
        if cuisine:
            items = [r for r in items if _loose_string_match(cuisine, r.cuisine)]

        if not items and pool:
            return _empty_with_hint(
                f"location='{location}', cuisine='{cuisine}'",
                "location/cuisine do substring/token match; try a broader regional term, "
                "drop cuisine, or use a different cuisine spelling",
            )

        if sort_by == "rating":
            items.sort(key=lambda r: (r.rating or 0), reverse=True)
        elif sort_by == "price_asc":
            items.sort(key=lambda r: _price_rank(r.price_range))
        elif sort_by == "price_desc":
            items.sort(key=lambda r: _price_rank(r.price_range), reverse=True)
        # relevance: keep insertion / query-match order

        results = [
            {"restaurant_id": r.restaurant_id, "name": r.name, "cuisine": r.cuisine,
             "location": r.location, "price_range": r.price_range,
             "seating_style": r.seating_style, "rating": r.rating,
             "attributes": r.attributes}
            for r in items
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.WRITE)
    def book_restaurant(
        self,
        restaurant_id: str,
        date_time: str,
        party_size: int,
        contact_info: str = "",
    ) -> dict:
        """Book a reservation at a selected restaurant.

        Args:
            restaurant_id: The restaurant to book.
            date_time: Reservation time.
            party_size: Number of diners.
            contact_info: Optional; defaults to persona.default_contact.
        """
        if restaurant_id not in self.db.restaurants:
            raise ValueError(f"Restaurant not found: {restaurant_id}")
        date_time = parse_iso_datetime(date_time, "date_time")
        if not contact_info:
            contact_info = self.db.persona.default_contact
        seq = len(self.db.restaurant_reservations) + 1
        res_id = f"RSV_R_{seq:03d}"
        self.db.restaurant_reservations[res_id] = RestaurantReservation(
            reservation_id=res_id, restaurant_id=restaurant_id,
            date_time=date_time, party_size=party_size,
            contact_info=contact_info,
        )
        return {"reservation_id": res_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def modify_restaurant_reservation(
        self, reservation_id: str,
        field: Literal["date_time", "party_size", "contact_info"],
        new_value: str,
    ) -> dict:
        """Modify an existing restaurant reservation.

        Only fields in _MODIFIABLE_FIELDS["modify_restaurant_reservation"]
        may be changed; status goes through cancel_restaurant_reservation.
        """
        allowed = self._MODIFIABLE_FIELDS.get("modify_restaurant_reservation", set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable on a restaurant reservation. "
                f"Allowed: {sorted(allowed)}"
            )
        res = self.db.restaurant_reservations.get(reservation_id)
        if not res:
            raise ValueError(f"Reservation not found: {reservation_id}")
        if field == "date_time":
            new_value = parse_iso_datetime(new_value, "new_value")
        elif field == "party_size":
            new_value = parse_int(new_value, "new_value")
        setattr(res, field, new_value)
        return {"reservation_id": reservation_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def cancel_restaurant_reservation(self, reservation_id: str) -> dict:
        """Cancel an existing restaurant reservation."""
        res = self.db.restaurant_reservations.get(reservation_id)
        if not res:
            raise ValueError(f"Reservation not found: {reservation_id}")
        res.status = "cancelled"
        return {"reservation_id": reservation_id, "status": "cancelled"}

    # -------- Events --------

    @is_tool(ToolType.READ)
    def search_events(
        self,
        location: Optional[str] = None,
        event_tags: Optional[list[str]] = None,
    ) -> dict:
        """Search local events or ticketed activities. Inspect each returned
        event's `date_time` to pick which one matches the instruction's
        time description.

        event_tags is episode-constrained to a closed set via tool_constraints.
        Filtering is by tag intersection — an event matches if ANY of its
        tags is in the query's tag list.
        """
        pool = list(self.db.events.values())
        items = pool
        if location:
            items = [e for e in items if _loose_string_match(location, e.location)]
        if event_tags:
            tset = {t.lower() for t in event_tags}
            items = [e for e in items
                     if any(t.lower() in tset for t in e.event_tags)]
        if not items and pool:
            return _empty_with_hint(
                f"location='{location}', event_tags={event_tags}",
                "filters do substring/token match against event.location and .event_tags; "
                "try omitting one filter or using a broader regional term",
            )
        results = [
            {"event_id": e.event_id, "name": e.name,
             "location": e.location, "date_time": e.date_time, "price": e.price,
             "event_tags": e.event_tags, "attributes": e.attributes}
            for e in items
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.WRITE)
    def book_event_ticket(
        self, event_id: str,
        ticket_type: Literal["general_admission", "senior", "vip", "child"],
        ticket_count: int,
        contact_info: str = "",
    ) -> dict:
        """Purchase or reserve tickets for a selected event."""
        if event_id not in self.db.events:
            raise ValueError(f"Event not found: {event_id}")
        parse_enum(ticket_type, {"general_admission", "senior", "vip", "child"}, "ticket_type")
        if not contact_info:
            contact_info = self.db.persona.default_contact
        seq = len(self.db.event_bookings) + 1
        booking_id = f"BKG_E_{seq:03d}"
        self.db.event_bookings[booking_id] = EventBooking(
            booking_id=booking_id, event_id=event_id,
            ticket_type=ticket_type, ticket_count=ticket_count,
            contact_info=contact_info,
        )
        return {"booking_id": booking_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def modify_event_ticket(
        self, booking_id: str,
        field: Literal["ticket_type", "ticket_count", "contact_info"],
        new_value: str,
    ) -> dict:
        """Modify an existing event ticket booking.

        Only fields in _MODIFIABLE_FIELDS["modify_event_ticket"] may be
        changed; status goes through cancel_event_ticket.
        """
        allowed = self._MODIFIABLE_FIELDS.get("modify_event_ticket", set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable on an event ticket booking. "
                f"Allowed: {sorted(allowed)}"
            )
        b = self.db.event_bookings.get(booking_id)
        if not b:
            raise ValueError(f"Booking not found: {booking_id}")
        if field == "ticket_count":
            new_value = parse_int(new_value, "new_value")
        setattr(b, field, new_value)
        return {"booking_id": booking_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def cancel_event_ticket(self, booking_id: str) -> dict:
        """Cancel an existing event ticket booking."""
        b = self.db.event_bookings.get(booking_id)
        if not b:
            raise ValueError(f"Booking not found: {booking_id}")
        b.status = "cancelled"
        return {"booking_id": booking_id, "status": "cancelled"}

    # -------- Service providers --------

    @is_tool(ToolType.READ)
    def search_service_providers(
        self,
        service_type: str,
        location: str,
        sort_by: ServiceSortBy = "relevance",
    ) -> dict:
        """Search local service providers that can be booked by appointment.
        Inspect each provider's available slots at booking time.
        """
        pool = list(self.db.providers.values())
        items = [p for p in pool if _loose_string_match(service_type, p.service_type)]
        if location:
            items = [p for p in items if _loose_string_match(location, p.location)]
        if not items and pool:
            return _empty_with_hint(
                f"service_type='{service_type}', location='{location}'",
                "filters do substring/token match against provider.service_type and .location; "
                "try a broader service_type term or different location spelling",
            )
        if sort_by == "distance":
            items.sort(key=lambda p: p.distance_km if p.distance_km is not None else float("inf"))
        elif sort_by == "rating":
            items.sort(key=lambda p: (p.rating or 0), reverse=True)
        results = [
            {"provider_id": p.provider_id, "name": p.name, "service_type": p.service_type,
             "location": p.location, "rating": p.rating,
             "distance_km": p.distance_km, "attributes": p.attributes}
            for p in items
        ]
        return {"count": len(results), "results": results}

    @is_tool(ToolType.WRITE)
    def book_service_appointment(
        self, provider_id: str, date_time: str,
        contact_info: str = "",
    ) -> dict:
        """Book an appointment with a selected local service provider."""
        provider = self.db.providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider not found: {provider_id}")
        date_time = parse_iso_datetime(date_time, "date_time")
        if not contact_info:
            contact_info = self.db.persona.default_contact
        seq = len(self.db.service_appointments) + 1
        appt_id = f"APT_{seq:03d}"
        self.db.service_appointments[appt_id] = ServiceAppointment(
            appointment_id=appt_id, provider_id=provider_id,
            service_type=provider.service_type, date_time=date_time,
            contact_info=contact_info,
        )
        return {"appointment_id": appt_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def modify_service_appointment(
        self, appointment_id: str,
        field: Literal["date_time", "contact_info"],
        new_value: str,
    ) -> dict:
        """Modify an existing local service appointment.

        Only fields in _MODIFIABLE_FIELDS["modify_service_appointment"] may
        be changed; status goes through cancel_service_appointment.
        """
        allowed = self._MODIFIABLE_FIELDS.get("modify_service_appointment", set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable on a service appointment. "
                f"Allowed: {sorted(allowed)}"
            )
        a = self.db.service_appointments.get(appointment_id)
        if not a:
            raise ValueError(f"Appointment not found: {appointment_id}")
        if field == "date_time":
            new_value = parse_iso_datetime(new_value, "new_value")
        setattr(a, field, new_value)
        return {"appointment_id": appointment_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def cancel_service_appointment(self, appointment_id: str) -> dict:
        """Cancel an existing local service appointment."""
        a = self.db.service_appointments.get(appointment_id)
        if not a:
            raise ValueError(f"Appointment not found: {appointment_id}")
        a.status = "cancelled"
        return {"appointment_id": appointment_id, "status": "cancelled"}

    @is_tool(ToolType.READ)
    def track_reservation_updates(
        self,
        reservation_id: Optional[str] = None,
        booking_id: Optional[str] = None,
        appointment_id: Optional[str] = None,
        update_scope: TrackingScope = "critical",
    ) -> dict:
        """Track updates on an existing restaurant reservation, event booking,
        or service appointment — surfaces venue changes, schedule shifts, and
        cancellations. Shape guard (env-enforced) — exactly one of
        reservation_id / booking_id / appointment_id must be provided.
        """
        if update_scope is not None:
            parse_enum(update_scope, {"critical", "all", "schedule_change"}, "update_scope")
        if reservation_id is not None:
            if reservation_id not in self.db.restaurant_reservations:
                raise ValueError(f"Reservation not found: {reservation_id}")
            target = reservation_id
            target_kind = "restaurant_reservation"
        elif booking_id is not None:
            if booking_id not in self.db.event_bookings:
                raise ValueError(f"Booking not found: {booking_id}")
            target = booking_id
            target_kind = "event_booking"
        elif appointment_id is not None:
            if appointment_id not in self.db.service_appointments:
                raise ValueError(f"Appointment not found: {appointment_id}")
            target = appointment_id
            target_kind = "service_appointment"
        else:
            # Shape guard catches this earlier; defensive fallback.
            raise ValueError(
                "Exactly one of reservation_id / booking_id / appointment_id required"
            )
        return {
            "tracking_id": f"TRK_{target}",
            "target": target,
            "target_kind": target_kind,
            "scope": update_scope,
            "status": "tracking_enabled",
        }


_PRICE_RANK = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


def _price_rank(price_range: Optional[str]) -> int:
    """Map restaurant price_range symbol to an ordinal. Unknown -> sentinel."""
    if not price_range:
        return 99
    return _PRICE_RANK.get(price_range.strip(), 99)


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(
    session_task,
    persona: PersonaProfile,
    allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    db = ReservationDB.from_references(persona, session_task.local_env.references)
    toolkit = ReservationTools(db)
    return ATREnv(
        domain="reservation", toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------


def _build_restaurant_row(ref_id, attrs, persona, spec):
    return {
        "restaurant_id": ref_id,
        "name": attrs.get("name", ref_id),
        "cuisine": attrs.get("cuisine", "unknown"),
        "location": attrs.get("location", ""),
        "rating": attrs.get("rating"),
        "price_range": attrs.get("price_range"),
        "seating_style": attrs.get("seating_style"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "cuisine", "location", "rating",
                                     "price_range", "seating_style")},
    }


def _build_event_row(ref_id, attrs, persona, spec):
    merged_tags = list(attrs.get("event_tags", []))
    legacy_type = attrs.get("event_type")
    if legacy_type and legacy_type not in merged_tags:
        merged_tags.append(legacy_type)
    return {
        "event_id": ref_id,
        "name": attrs.get("name", ref_id),
        "location": attrs.get("location", ""),
        "date_time": attrs.get("date_time"),
        "price": attrs.get("price"),
        "event_tags": merged_tags,
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "event_type", "location",
                                     "date_time", "price", "event_tags")},
    }


def _build_provider_row(ref_id, attrs, persona, spec):
    return {
        "provider_id": ref_id,
        "name": attrs.get("name", ref_id),
        "service_type": attrs.get("service_type", "other"),
        "location": attrs.get("location", ""),
        "rating": attrs.get("rating"),
        "distance_km": attrs.get("distance_km"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "service_type", "location",
                                     "rating", "distance_km")},
    }


def _derive_reservation_row(source_ref_id, source_attrs, persona, spec):
    return {
        "reservation_id": source_attrs["reservation_id"],
        "restaurant_id": source_ref_id,
        "date_time": source_attrs.get("reservation_date_time", ""),
        "party_size": int(source_attrs.get("party_size", 1)),
        "contact_info": persona.default_contact,
        "status": source_attrs.get("reservation_status", "confirmed"),
    }


def _derive_event_booking_row(source_ref_id, source_attrs, persona, spec):
    return {
        "booking_id": source_attrs["booking_id"],
        "event_id": source_ref_id,
        "ticket_type": source_attrs.get("ticket_type", "standard"),
        "ticket_count": int(source_attrs.get("ticket_count", 1)),
        "contact_info": persona.default_contact,
        "status": source_attrs.get("booking_status", "confirmed"),
    }


def _derive_appointment_row(source_ref_id, source_attrs, persona, spec):
    return {
        "appointment_id": source_attrs["appointment_id"],
        "provider_id": source_ref_id,
        "service_type": source_attrs.get("service_type", "other"),
        "date_time": source_attrs.get("appointment_date_time", ""),
        "contact_info": persona.default_contact,
        "status": source_attrs.get("appointment_status", "confirmed"),
    }


_RESERVATION_TABLES = [
    TableSpec(
        name="restaurants", model=Restaurant, kind="primary",
        source_ref_type="restaurant",
        promoted_attrs=["name", "cuisine", "location", "rating",
                        "price_range", "seating_style"],
        operating_tools=["search_restaurants", "book_restaurant"],
        discovery_tools=["search_restaurants"],
        build_row=_build_restaurant_row,
    ),
    TableSpec(
        name="events", model=EventRecord, kind="primary",
        source_ref_type="event",
        promoted_attrs=["name", "location", "date_time", "price", "event_tags"],
        operating_tools=["search_events", "book_event_ticket"],
        discovery_tools=["search_events"],
        build_row=_build_event_row,
    ),
    TableSpec(
        name="providers", model=ServiceProvider, kind="primary",
        source_ref_type="service_provider",
        promoted_attrs=["name", "service_type", "location", "rating", "distance_km"],
        operating_tools=["search_service_providers", "book_service_appointment"],
        discovery_tools=["search_service_providers"],
        build_row=_build_provider_row,
    ),
    TableSpec(
        name="restaurant_reservations", model=RestaurantReservation, kind="derived",
        source_ref_type="restaurant", source_attr="reservation_id",
        operating_tools=["modify_restaurant_reservation",
                         "cancel_restaurant_reservation"],
        discovery_tools=["search_restaurants"],
        derive_row=_derive_reservation_row,
    ),
    TableSpec(
        name="event_bookings", model=EventBooking, kind="derived",
        source_ref_type="event", source_attr="booking_id",
        operating_tools=["modify_event_ticket", "cancel_event_ticket"],
        discovery_tools=["search_events"],
        derive_row=_derive_event_booking_row,
    ),
    TableSpec(
        name="service_appointments", model=ServiceAppointment, kind="derived",
        source_ref_type="service_provider", source_attr="appointment_id",
        operating_tools=["modify_service_appointment",
                         "cancel_service_appointment"],
        discovery_tools=["search_service_providers"],
        derive_row=_derive_appointment_row,
    ),
]

ReservationDB._TABLES = _RESERVATION_TABLES
register_domain_tables("reservation", _RESERVATION_TABLES)
