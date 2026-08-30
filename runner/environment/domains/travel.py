"""Travel domain: destinations, flights, hotels, ground transport, tracking.

Write tools auto-fill identity fields (traveler_info, guest_info, payment_method)
from persona.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from runner.environment.base import (
    ATRDB, ATREnv, ATRToolKitBase, PersonaProfile, ToolType,
    _empty_with_hint, _loose_string_match, is_tool,
)
from runner.environment.tables import TableSpec, register_domain_tables
from runner.environment._validators import parse_int, parse_enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Destination(BaseModel):
    destination_id: str
    name: str
    region: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class FlightOffer(BaseModel):
    flight_offer_id: str
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    airline: Optional[str] = None
    price: Optional[float] = None
    available_seats: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class HotelOffer(BaseModel):
    hotel_offer_id: str
    name: str
    location: str
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    available_rooms: Optional[int] = None
    max_guests_per_room: Optional[int] = None
    price_per_night: Optional[float] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TransportOffer(BaseModel):
    transport_offer_id: str
    origin: str
    destination: str
    mode: str  # train / bus / car_transfer
    departure_date: str
    price: Optional[float] = None
    available_seats: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TripStop(BaseModel):
    stop_id: str
    name: str
    location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class TravelBooking(BaseModel):
    booking_id: str
    kind: Literal["flight", "hotel", "ground_transport"]
    offer_id: str
    payment_method: str
    status: str = "confirmed"
    traveler_info: Optional[str] = None
    guest_info: Optional[str] = None
    room_count: Optional[int] = None
    seat_preference: Optional[str] = None  # flights: window|aisle|middle


class TripPlan(BaseModel):
    trip_id: str
    destination: str
    duration_days: Optional[int] = None
    traveler_count: Optional[int] = None
    # v0.12: agent commits to specific trip-stop selections via this list.
    # Rules about which kind of stop to prioritize land here (as ids).
    selected_stop_ids: list[str] = Field(default_factory=list)
    include_daily_breakdown: bool = False
    include_stopovers: bool = False
    include_budget_summary: bool = False


class DestinationShortlist(BaseModel):
    """v0.12: agent-produced shortlist of recommended destinations. This is
    the execute anchor for "recommend destinations" style rules — gold lands
    on destination_ids here, not on search_destinations filters.
    """
    shortlist_id: str
    destination_ids: list[str] = Field(default_factory=list)


class TravelDB(ATRDB):
    destinations: dict[str, Destination] = Field(default_factory=dict)
    flight_offers: dict[str, FlightOffer] = Field(default_factory=dict)
    hotel_offers: dict[str, HotelOffer] = Field(default_factory=dict)
    transport_offers: dict[str, TransportOffer] = Field(default_factory=dict)
    trip_stops: dict[str, TripStop] = Field(default_factory=dict)
    trips: dict[str, TripPlan] = Field(default_factory=dict)
    bookings: dict[str, TravelBooking] = Field(default_factory=dict)
    shortlists: dict[str, DestinationShortlist] = Field(default_factory=dict)

    _KNOWN_REF_TYPES = {
        "destination", "flight_offer", "hotel_offer", "transport_offer", "trip_stop",
    }

    _TABLES: list[TableSpec] = []

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

UpdateScope = Literal["critical", "all", "price_change", "schedule_change"]
SeatPreference = Literal["window", "aisle", "middle"]
OfferSortBy = Literal["relevance", "price_asc", "price_desc"]


class TravelTools(ATRToolKitBase):
    db: TravelDB
    _shape_guards = {
        "track_trip_updates": {"exactly_one_of": ["trip_id", "booking_id"]},
    }

    # Fields modify_* tools are allowed to write, keyed by tool name.
    # Kind-specific: e.g. flight bookings can change seat preference but
    # not room_count (hotel-only). Rejects bypass via the generic
    # (field,new_value) signature.
    _MODIFIABLE_FIELDS: dict[str, set[str]] = {
        "modify_flight_booking": {"seat_preference", "traveler_info"},
        "modify_hotel_booking": {"room_count", "guest_info"},
    }

    def __init__(self, db: TravelDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def search_destinations(
        self,
        destination_constraints: Optional[list[str]] = None,
    ) -> dict:
        """Search destination candidates or route anchors for a trip idea.
        Filter via destination_constraints (matched substring across name,
        location, and tag fields).
        """
        pool = list(self.db.destinations.values())
        items = pool
        if destination_constraints:
            cs = [c.lower() for c in destination_constraints]
            items = [d for d in items
                     if any(_loose_string_match(c, d.region or "")
                            or _loose_string_match(c, " ".join(d.tags))
                            or _loose_string_match(c, d.name) for c in cs)]
        if not items and pool:
            return _empty_with_hint(
                f"destination_constraints={destination_constraints}",
                "try fewer / broader constraints, or omit destination_constraints to see all destinations in the session pool",
            )
        return {"count": len(items), "results": [d.model_dump() for d in items]}

    @is_tool(ToolType.WRITE)
    def shortlist_destinations(
        self,
        destination_ids: list[str],
    ) -> dict:
        """Commit a shortlist of recommended destinations for the user.
        This is the id-level decision anchor for destination-recommendation
        rules — gold lands here, not on search_destinations filters.
        Each id must already exist in the DB.
        """
        for did in destination_ids:
            if did not in self.db.destinations:
                raise ValueError(f"Destination not found: {did}")
        seq = len(self.db.shortlists) + 1
        shortlist_id = f"DSH_{seq:03d}"
        self.db.shortlists[shortlist_id] = DestinationShortlist(
            shortlist_id=shortlist_id,
            destination_ids=list(destination_ids),
        )
        return {
            "shortlist_id": shortlist_id,
            "destination_ids": list(destination_ids),
            "count": len(destination_ids),
        }

    @is_tool(ToolType.WRITE)
    def plan_trip(
        self,
        destination: str,
        duration_days: Optional[int] = None,
        traveler_count: Optional[int] = None,
        selected_stop_ids: Optional[list[str]] = None,
        include_daily_breakdown: bool = False,
        include_stopovers: bool = False,
        include_budget_summary: bool = False,
    ) -> dict:
        """Build a draft itinerary or trip plan. When the agent wants to
        commit to specific stops in the itinerary, pass their ids via
        selected_stop_ids — each id must already exist in the DB (from a
        prior search_trip_stops call or preseeded references).

        Classified as WRITE (v0.15): the call mutates DB state by creating
        a TripPlan record discoverable via its trip_id in later calls
        (track_trip_updates, replan_trip). ToolType=READ would falsely
        imply idempotent read-only semantics under τ²-style replay.
        """
        selected = selected_stop_ids or []
        for sid in selected:
            if sid not in self.db.trip_stops:
                raise ValueError(f"Trip stop not found: {sid}")
        seq = len(self.db.trips) + 1
        trip_id = f"TRP_{seq:03d}"
        self.db.trips[trip_id] = TripPlan(
            trip_id=trip_id, destination=destination,
            duration_days=duration_days, traveler_count=traveler_count,
            selected_stop_ids=list(selected),
            include_daily_breakdown=include_daily_breakdown,
            include_stopovers=include_stopovers,
            include_budget_summary=include_budget_summary,
        )
        return {
            "trip_id": trip_id, "destination": destination,
            "selected_stop_ids": list(selected),
            "duration_days": duration_days,
            "sections": {
                "daily_breakdown": include_daily_breakdown,
                "stopovers": include_stopovers,
                "budget_summary": include_budget_summary,
            },
        }

    @is_tool(ToolType.READ)
    def search_trip_stops(
        self,
        destination: str = "",
        stop_tags: Optional[list[str]] = None,
    ) -> dict:
        """Search stops, route anchors, or points of interest for a trip.

        destination matches loosely against each stop's name / location
        (substring, case-insensitive) — stops whose location or name mentions
        the destination are kept. Pass an empty string (or omit) to see all
        stops in the session pool — useful when a regional term doesn't
        match any seeded stop. stop_tags filters by tag intersection.
        """
        pool = list(self.db.trip_stops.values())
        items = pool
        if destination:
            items = [s for s in items
                     if _loose_string_match(destination, s.name)
                     or _loose_string_match(destination, s.location or "")]
        if stop_tags:
            tset = [t.lower() for t in stop_tags]
            items = [s for s in items if any(t in [x.lower() for x in s.tags] for t in tset)]
        if not items and pool:
            return _empty_with_hint(
                f"destination='{destination}', stop_tags={stop_tags}",
                "destination filter does substring/token match against stop.location and stop.name; "
                "try a broader regional term (e.g. 'Greater Vancouver' instead of 'Vancouver'), "
                "omit destination to see all stops in the session pool, or relax stop_tags",
            )
        return {"count": len(items), "results": [s.model_dump() for s in items]}

    @is_tool(ToolType.READ)
    def search_flights(
        self, origin: str, destination: str,
        passenger_count: int,
        sort_by: OfferSortBy = "relevance",
    ) -> dict:
        """Search flight candidates. Inspect each returned offer's
        `departure_date` / `return_date` to pick the right one.

        sort_by defaults to 'relevance' (catalog order). Use 'price_asc' /
        'price_desc' for explicit price ordering — offers with missing price
        are placed last.
        """
        pool = list(self.db.flight_offers.values())
        items = [f for f in pool
                 if _loose_string_match(origin, f.origin)
                 and _loose_string_match(destination, f.destination)
                 and (f.available_seats is None or passenger_count <= f.available_seats)]
        items = _sort_offers(items, sort_by, price_attr="price")
        if not items and pool:
            return _empty_with_hint(
                f"origin='{origin}', destination='{destination}'",
                "origin / destination do substring/token match; try airport names instead of city codes (or vice versa)",
            )
        return {"count": len(items), "results": [f.model_dump() for f in items]}

    @is_tool(ToolType.READ)
    def search_hotels(
        self, location: str,
        guest_count: int, room_count: int,
        sort_by: OfferSortBy = "relevance",
    ) -> dict:
        """Search hotel or lodging candidates. Inspect each returned offer's
        `check_in_date` / `check_out_date` to pick the right one.

        sort_by defaults to 'relevance' (catalog order). 'price_asc' /
        'price_desc' sort by price_per_night.
        """
        pool = list(self.db.hotel_offers.values())
        items = [h for h in pool
                 if _loose_string_match(location, h.location)
                 and (h.available_rooms is None or room_count <= h.available_rooms)
                 and (h.max_guests_per_room is None
                      or guest_count <= h.max_guests_per_room * room_count)]
        items = _sort_offers(items, sort_by, price_attr="price_per_night")
        if not items and pool:
            return _empty_with_hint(
                f"location='{location}'",
                "location does substring/token match against hotel.location; "
                "try a broader regional term",
            )
        return {"count": len(items), "results": [h.model_dump() for h in items]}

    @is_tool(ToolType.READ)
    def search_ground_transport(
        self, origin: str, destination: str, passenger_count: int,
        sort_by: OfferSortBy = "relevance",
    ) -> dict:
        """Search train, bus, or car-transfer candidates for a trip.
        Inspect each returned offer's `departure_date` to pick the right one.

        sort_by defaults to 'relevance' (catalog order); 'price_asc' /
        'price_desc' sort by ticket price.
        """
        pool = list(self.db.transport_offers.values())
        items = [t for t in pool
                 if _loose_string_match(origin, t.origin)
                 and _loose_string_match(destination, t.destination)
                 and (t.available_seats is None or passenger_count <= t.available_seats)]
        items = _sort_offers(items, sort_by, price_attr="price")
        if not items and pool:
            return _empty_with_hint(
                f"origin='{origin}', destination='{destination}'",
                "origin / destination do substring/token match; try a broader regional term",
            )
        return {"count": len(items), "results": [t.model_dump() for t in items]}

    @is_tool(ToolType.WRITE)
    def book_flight(
        self, flight_offer_id: str, traveler_info: str = "",
        seat_preference: Optional[SeatPreference] = None,
        payment_method: str = "",
    ) -> dict:
        """Book a selected flight offer. seat_preference is persisted on the
        booking so downstream modify/cancel calls observe it.
        """
        if flight_offer_id not in self.db.flight_offers:
            raise ValueError(f"Flight not found: {flight_offer_id}")
        if seat_preference is not None:
            parse_enum(seat_preference, {"window", "aisle", "middle"}, "seat_preference")
        if not traveler_info:
            traveler_info = self.db.persona.default_contact
        if not payment_method:
            payment_method = self.db.persona.default_payment_method
        seq = len(self.db.bookings) + 1
        booking_id = f"FLT_{seq:03d}"
        self.db.bookings[booking_id] = TravelBooking(
            booking_id=booking_id, kind="flight", offer_id=flight_offer_id,
            payment_method=payment_method, traveler_info=traveler_info,
            seat_preference=seat_preference,
        )
        return {"booking_id": booking_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def book_hotel(
        self, hotel_offer_id: str, guest_info: str = "",
        payment_method: str = "",
    ) -> dict:
        """Book a selected hotel offer. Room count is encoded by hotel_offer_id
        (set at search_hotels time)."""
        offer = self.db.hotel_offers.get(hotel_offer_id)
        if offer is None:
            raise ValueError(f"Hotel not found: {hotel_offer_id}")
        if not guest_info:
            guest_info = self.db.persona.default_contact
        if not payment_method:
            payment_method = self.db.persona.default_payment_method
        seq = len(self.db.bookings) + 1
        booking_id = f"HTL_{seq:03d}"
        self.db.bookings[booking_id] = TravelBooking(
            booking_id=booking_id, kind="hotel", offer_id=hotel_offer_id,
            payment_method=payment_method, guest_info=guest_info,
            room_count=getattr(offer, "room_count", 1),
        )
        return {"booking_id": booking_id, "status": "confirmed"}

    @is_tool(ToolType.WRITE)
    def book_ground_transport(
        self, transport_offer_id: str, traveler_info: str = "",
        payment_method: str = "",
    ) -> dict:
        """Book a selected train, bus, or car-transfer offer."""
        if transport_offer_id not in self.db.transport_offers:
            raise ValueError(f"Transport offer not found: {transport_offer_id}")
        if not traveler_info:
            traveler_info = self.db.persona.default_contact
        if not payment_method:
            payment_method = self.db.persona.default_payment_method
        seq = len(self.db.bookings) + 1
        booking_id = f"GND_{seq:03d}"
        self.db.bookings[booking_id] = TravelBooking(
            booking_id=booking_id, kind="ground_transport", offer_id=transport_offer_id,
            payment_method=payment_method, traveler_info=traveler_info,
        )
        return {"booking_id": booking_id, "status": "confirmed"}

    def _modify_booking(
        self, tool_name: str, booking_id: str, field: str, new_value: str,
    ) -> dict:
        allowed = self._MODIFIABLE_FIELDS.get(tool_name, set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable via {tool_name}. "
                f"Allowed: {sorted(allowed)}"
            )
        b = self.db.bookings.get(booking_id)
        if not b:
            raise ValueError(f"Booking not found: {booking_id}")
        if field == "room_count":
            new_value = parse_int(new_value, "new_value")
        setattr(b, field, new_value)
        return {"booking_id": booking_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def modify_flight_booking(
        self, booking_id: str,
        field: Literal["seat_preference", "traveler_info"],
        new_value: str,
    ) -> dict:
        """Modify an existing flight booking."""
        return self._modify_booking(
            "modify_flight_booking", booking_id, field, new_value,
        )

    @is_tool(ToolType.WRITE)
    def modify_hotel_booking(
        self, booking_id: str,
        field: Literal["room_count", "guest_info"],
        new_value: str,
    ) -> dict:
        """Modify an existing hotel booking."""
        return self._modify_booking(
            "modify_hotel_booking", booking_id, field, new_value,
        )

    def _cancel_booking(self, booking_id: str) -> dict:
        b = self.db.bookings.get(booking_id)
        if not b:
            raise ValueError(f"Booking not found: {booking_id}")
        b.status = "cancelled"
        return {"booking_id": booking_id, "status": "cancelled"}

    @is_tool(ToolType.WRITE)
    def cancel_flight_booking(self, booking_id: str) -> dict:
        """Cancel an existing flight booking."""
        return self._cancel_booking(booking_id)

    @is_tool(ToolType.WRITE)
    def cancel_hotel_booking(self, booking_id: str) -> dict:
        """Cancel an existing hotel booking."""
        return self._cancel_booking(booking_id)

    @is_tool(ToolType.WRITE)
    def cancel_ground_transport_booking(self, booking_id: str) -> dict:
        """Cancel an existing train, bus, or car-transfer booking."""
        return self._cancel_booking(booking_id)

    @is_tool(ToolType.READ)
    def track_trip_updates(
        self,
        trip_id: Optional[str] = None,
        booking_id: Optional[str] = None,
        update_scope: UpdateScope = "critical",
    ) -> dict:
        """Track important updates for an itinerary, route, or booked trip component.
        Shape guard (env-enforced) — exactly one of trip_id / booking_id must be
        provided; that id must come from a prior plan_trip / book_* call.
        """
        if trip_id is not None:
            if trip_id not in self.db.trips:
                raise ValueError(f"Trip not found: {trip_id}")
            target = trip_id
            target_kind = "trip"
        elif booking_id is not None:
            if booking_id not in self.db.bookings:
                raise ValueError(f"Booking not found: {booking_id}")
            target = booking_id
            target_kind = "booking"
        else:
            raise ValueError("Exactly one of trip_id / booking_id required")
        parse_enum(update_scope, {"critical", "all", "price_change", "schedule_change"}, "update_scope")
        return {
            "tracking_id": f"TRK_{target}",
            "target": target,
            "target_kind": target_kind,
            "scope": update_scope,
            "status": "tracking_enabled",
        }

    @is_tool(ToolType.WRITE)
    def replan_trip(
        self, trip_id: str, replanning_goal: str,
        disrupted_item_id: Optional[str] = None,
        selected_alternative_stop_ids: Optional[list[str]] = None,
        replacement_type: Optional[Literal["swap_single_stop", "swap_all_stops"]] = None,
    ) -> dict:
        """Adjust a trip plan after disruption.

        Args:
            trip_id: The trip being replanned.
            replanning_goal: Free-text description of what should change.
            disrupted_item_id: Optional id of the original booking, offer, or
                existing stop on the trip that fell through. Must reference a
                known booking, flight/hotel/ground offer, or trip_stop with a
                matching trip_id in this session's DB when supplied.
            selected_alternative_stop_ids: Optional ids of replacement stops
                the agent commits to. Mirrors plan_trip.selected_stop_ids —
                this is the id-level decision hook for rules about replacement
                selection (e.g. "same-theme substitutes"). Each id must exist
                in the session's trip_stops catalog.
            replacement_type: Free-text classification of the replacement.
        """
        trip = self.db.trips.get(trip_id)
        if not trip:
            raise ValueError(f"Trip not found: {trip_id}")
        if replacement_type is not None:
            parse_enum(replacement_type, {"swap_single_stop", "swap_all_stops"}, "replacement_type")
        if disrupted_item_id is not None:
            known = (
                disrupted_item_id in self.db.bookings
                or disrupted_item_id in self.db.flight_offers
                or disrupted_item_id in self.db.hotel_offers
                or disrupted_item_id in self.db.transport_offers
            )
            disrupted_stop = self.db.trip_stops.get(disrupted_item_id)
            if disrupted_stop is not None:
                stop_trip_id = disrupted_stop.attributes.get("trip_id")
                if stop_trip_id == trip_id:
                    known = True
                elif not known:
                    raise ValueError(
                        f"Disrupted stop {disrupted_item_id} belongs to trip "
                        f"{stop_trip_id!r}, not {trip_id!r}"
                    )
            if not known:
                raise ValueError(f"Disrupted item not found: {disrupted_item_id}")
        if selected_alternative_stop_ids:
            for sid in selected_alternative_stop_ids:
                if sid not in self.db.trip_stops:
                    raise ValueError(f"Alternative stop not found: {sid}")
        return {
            "trip_id": trip_id,
            "replanning_goal": replanning_goal,
            "disrupted_item_id": disrupted_item_id,
            "selected_alternative_stop_ids": selected_alternative_stop_ids or [],
            "replacement_type": replacement_type,
            "status": "replanned",
        }


def _sort_offers(items: list, sort_by: str, price_attr: str) -> list:
    """Sort offers by price when requested; None prices sort to the end.

    Kept stable with a secondary key (offer id) so the same query always
    returns the same order regardless of dict iteration order.
    """
    if sort_by == "price_asc":
        return sorted(
            items,
            key=lambda x: (getattr(x, price_attr) is None,
                           getattr(x, price_attr) or 0,
                           getattr(x, _id_attr(x))),
        )
    if sort_by == "price_desc":
        return sorted(
            items,
            key=lambda x: (getattr(x, price_attr) is None,
                           -(getattr(x, price_attr) or 0),
                           getattr(x, _id_attr(x))),
        )
    return items


def _id_attr(obj) -> str:
    for name in ("flight_offer_id", "hotel_offer_id", "transport_offer_id"):
        if hasattr(obj, name):
            return name
    return "destination_id"


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(
    session_task, persona: PersonaProfile, allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    db = TravelDB.from_references(persona, session_task.local_env.references)
    toolkit = TravelTools(db)
    return ATREnv(
        domain="travel", toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------


def _build_destination_row(ref_id, attrs, persona, spec):
    return {
        "destination_id": ref_id,
        "name": attrs.get("name", ref_id),
        "region": attrs.get("region"),
        "tags": attrs.get("tags", []),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "region", "tags")},
    }


_FLIGHT_STRUCTURED = {
    "origin", "destination", "departure_date", "return_date",
    "airline", "price", "available_seats",
}


def _build_flight_row(ref_id, attrs, persona, spec):
    return {
        "flight_offer_id": ref_id,
        "origin": attrs.get("origin", ""),
        "destination": attrs.get("destination", ""),
        "departure_date": attrs.get("departure_date", ""),
        "return_date": attrs.get("return_date"),
        "airline": attrs.get("airline"),
        "price": attrs.get("price"),
        "available_seats": attrs.get("available_seats"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _FLIGHT_STRUCTURED},
    }


_HOTEL_STRUCTURED = {
    "name", "location", "check_in_date", "check_out_date",
    "available_rooms", "max_guests_per_room", "price_per_night",
}


def _build_hotel_row(ref_id, attrs, persona, spec):
    return {
        "hotel_offer_id": ref_id,
        "name": attrs.get("name", ref_id),
        "location": attrs.get("location", ""),
        "check_in_date": attrs.get("check_in_date"),
        "check_out_date": attrs.get("check_out_date"),
        "available_rooms": attrs.get("available_rooms"),
        "max_guests_per_room": attrs.get("max_guests_per_room"),
        "price_per_night": attrs.get("price_per_night"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _HOTEL_STRUCTURED},
    }


_TRANSPORT_STRUCTURED = {
    "origin", "destination", "mode", "departure_date",
    "price", "available_seats",
}


def _build_transport_row(ref_id, attrs, persona, spec):
    return {
        "transport_offer_id": ref_id,
        "origin": attrs.get("origin", ""),
        "destination": attrs.get("destination", ""),
        "mode": attrs.get("mode", "train"),
        "departure_date": attrs.get("departure_date", ""),
        "price": attrs.get("price"),
        "available_seats": attrs.get("available_seats"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in _TRANSPORT_STRUCTURED},
    }


def _build_trip_stop_row(ref_id, attrs, persona, spec):
    merged_tags = list(attrs.get("tags", []))
    legacy_type = attrs.get("stop_type")
    if legacy_type and legacy_type not in merged_tags:
        merged_tags.append(legacy_type)
    return {
        "stop_id": ref_id,
        "name": attrs.get("name", ref_id),
        "location": attrs.get("location"),
        "tags": merged_tags,
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "location", "tags", "stop_type")},
    }


def _derive_trip_row(source_ref_id, source_attrs, persona, spec):
    return {
        "trip_id": source_attrs["trip_id"],
        "destination": source_attrs.get("location", ""),
        "selected_stop_ids": [source_ref_id],
    }


def _merge_trip_row(existing_trip, source_ref_id, source_attrs, persona):
    new_stops = list(existing_trip.selected_stop_ids)
    if source_ref_id not in new_stops:
        new_stops.append(source_ref_id)
    return existing_trip.model_copy(update={"selected_stop_ids": new_stops})


_TRAVEL_TABLES = [
    TableSpec(
        name="destinations", model=Destination, kind="primary",
        source_ref_type="destination",
        promoted_attrs=["name", "region", "tags"],
        operating_tools=["search_destinations", "shortlist_destinations"],
        discovery_tools=["search_destinations"],
        build_row=_build_destination_row,
    ),
    TableSpec(
        name="flight_offers", model=FlightOffer, kind="primary",
        source_ref_type="flight_offer",
        promoted_attrs=["origin", "destination", "departure_date", "return_date",
                        "airline", "price", "available_seats"],
        operating_tools=["search_flights", "book_flight"],
        discovery_tools=["search_flights"],
        build_row=_build_flight_row,
    ),
    TableSpec(
        name="hotel_offers", model=HotelOffer, kind="primary",
        source_ref_type="hotel_offer",
        promoted_attrs=["name", "location", "check_in_date", "check_out_date",
                        "available_rooms", "max_guests_per_room", "price_per_night"],
        operating_tools=["search_hotels", "book_hotel"],
        discovery_tools=["search_hotels"],
        build_row=_build_hotel_row,
    ),
    TableSpec(
        name="transport_offers", model=TransportOffer, kind="primary",
        source_ref_type="transport_offer",
        promoted_attrs=["origin", "destination", "mode", "departure_date",
                        "price", "available_seats"],
        operating_tools=["search_ground_transport", "book_ground_transport"],
        discovery_tools=["search_ground_transport"],
        build_row=_build_transport_row,
    ),
    TableSpec(
        name="trip_stops", model=TripStop, kind="primary",
        source_ref_type="trip_stop",
        promoted_attrs=["name", "location", "tags"],
        operating_tools=["search_trip_stops", "plan_trip"],
        discovery_tools=["search_trip_stops"],
        build_row=_build_trip_stop_row,
    ),
    TableSpec(
        name="trips", model=TripPlan, kind="derived",
        source_ref_type="trip_stop", source_attr="trip_id",
        operating_tools=["replan_trip", "track_trip_updates"],
        discovery_tools=["search_trip_stops"],
        derive_row=_derive_trip_row,
        merge_row=_merge_trip_row,
    ),
    TableSpec(
        name="bookings", model=TravelBooking, kind="runtime",
        operating_tools=[
            "modify_flight_booking", "modify_hotel_booking",
            "cancel_flight_booking", "cancel_hotel_booking",
            "cancel_ground_transport_booking",
        ],
        discovery_tools=[],
    ),
]

TravelDB._TABLES = _TRAVEL_TABLES
register_domain_tables("travel", _TRAVEL_TABLES)
