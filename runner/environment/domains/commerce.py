"""Commerce domain: products, orders, and shopping tools.

Data model:
  PersonaProfile    episode-level identity (address, payment)
  Product           catalog entry (lifted from session references)
  Order             created by place_order (session-local)

Tools:
  search_products   READ — search with sort_by ranking and category/budget filters
  compare_products  READ — side-by-side attribute comparison
  place_order       WRITE — create an order and return confirmation
"""
from __future__ import annotations

import re
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
from runner.environment._validators import parse_int, parse_iso_date, parse_enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """A product in the catalog. Lifted from references (type=='product')."""
    product_id: str
    name: str
    price: float
    category: str
    rating: Optional[float] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    order_id: str
    product_id: str
    quantity: int
    shipping_address: str
    payment_method: str
    total: float
    status: str = "confirmed"


class Subscription(BaseModel):
    subscription_id: str
    name: str
    plan: str
    status: str = "active"  # active / paused / cancelled
    pause_until: Optional[str] = None
    price_per_period: Optional[float] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class CommerceDB(ATRDB):
    persona: PersonaProfile
    products: dict[str, Product] = Field(default_factory=dict)
    orders: dict[str, Order] = Field(default_factory=dict)
    subscriptions: dict[str, Subscription] = Field(default_factory=dict)
    shopping_lists: dict[str, dict] = Field(default_factory=dict)

    _KNOWN_REF_TYPES = {"product", "subscription"}

    _TABLES: list[TableSpec] = []  # populated below after helper defs

    @classmethod
    def from_references(cls, persona, references):
        return cls.hydrate_all(persona, references)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

SortBy = Literal["relevance", "price_asc", "price_desc", "rating"]
ResolutionType = Literal["refund", "exchange", "store_credit"]
SubscriptionPlan = Literal["basic", "pro", "premium", "enterprise"]


class CommerceTools(ATRToolKitBase):
    """Commerce domain toolkit."""

    db: CommerceDB

    # Fields modify_* tools are allowed to write, keyed by tool name.
    # Any field outside this set is rejected at runtime — prevents the
    # agent from using the generic modify_(field,new_value) signature to
    # bypass normal flows (e.g. flipping order.status or order.total
    # directly instead of routing through cancel_order / return_order).
    _MODIFIABLE_FIELDS: dict[str, set[str]] = {
        "modify_order": {"quantity", "shipping_address"},
    }

    def __init__(self, db: CommerceDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def search_products(
        self,
        category: Optional[str] = None,
        budget: Optional[float] = None,
        sort_by: SortBy = "relevance",
    ) -> dict:
        """Search the product catalog.

        Returns a ranked list of products matching the filters. Each result
        carries the product's core decision fields plus its `attributes`
        dict (uniform surface policy: data-gen controls visibility by what
        it puts in reference.attributes).

        Args:
            category: Category filter (e.g. 'garden_plants', 'multivitamin').
            budget: Max price inclusive.
            sort_by: How to rank the results.
        """
        pool = list(self.db.products.values())
        items = pool
        if category:
            items = [p for p in items if _loose_string_match(category, p.category)]
        if budget is not None:
            items = [p for p in items if p.price <= budget]

        if not items and pool:
            return _empty_with_hint(
                f"category='{category}', budget={budget}",
                "category does substring/token match against product.category; "
                "try a broader category term, or omit category/budget to widen "
                "the search (omitting both returns the full catalog)",
            )

        if sort_by == "price_asc":
            items.sort(key=lambda p: p.price)
        elif sort_by == "price_desc":
            items.sort(key=lambda p: p.price, reverse=True)
        elif sort_by == "rating":
            items.sort(key=lambda p: (p.rating or 0), reverse=True)
        # relevance: leave query's token-matched order

        results = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "price": p.price,
                "category": p.category,
                "rating": p.rating,
                "attributes": p.attributes,
            }
            for p in items
        ]
        return {
            "count": len(items),
            "results": results,
        }

    @is_tool(ToolType.READ)
    def compare_products(
        self,
        product_ids: list[str],
    ) -> dict:
        """Compare several products side by side.

        Returns core decision fields plus the `attributes` dict (uniform
        surface policy). Rule-relevant attributes (sugar_free, origin,
        climate_fit, …) are visible when data-gen places them on the
        product reference.

        Args:
            product_ids: IDs of the products to compare.
        """
        comparison = []
        for pid in product_ids:
            p = self.db.products.get(pid)
            if p is None:
                comparison.append({"product_id": pid, "error": "not found"})
                continue
            comparison.append({
                "product_id": p.product_id,
                "name": p.name,
                "price": p.price,
                "category": p.category,
                "rating": p.rating,
                "attributes": p.attributes,
            })
        return {"comparison": comparison}

    @is_tool(ToolType.WRITE)
    def build_shopping_list(
        self, items: list[str],
        list_name: Optional[str] = None,
    ) -> dict:
        """Create or update a shopping list from selected items."""
        lid = f"LIST_{len(self.db.shopping_lists)+1:03d}"
        self.db.shopping_lists[lid] = {
            "list_id": lid, "name": list_name or lid, "items": items,
        }
        return {"list_id": lid, "item_count": len(items)}

    @is_tool(ToolType.WRITE)
    def place_order(
        self,
        product_id: str,
        quantity: int,
        shipping_address: str = "",
        payment_method: str = "",
    ) -> dict:
        """Place an order for one product. Shipping_address and payment_method
        default to persona values when empty; the response never echoes those
        identity fields back.
        """
        if product_id not in self.db.products:
            raise ValueError(f"Product not found: {product_id}")
        product = self.db.products[product_id]
        if not shipping_address:
            shipping_address = self.db.persona.default_shipping_address
        if not payment_method:
            payment_method = self.db.persona.default_payment_method
        seq = len(self.db.orders) + 1
        order_id = f"ORD_{product_id[:10]}_{seq:03d}"
        order = Order(
            order_id=order_id, product_id=product_id, quantity=quantity,
            shipping_address=shipping_address, payment_method=payment_method,
            total=round(product.price * quantity, 2), status="confirmed",
        )
        self.db.orders[order_id] = order
        return {
            "order_id": order_id, "product_id": product_id,
            "quantity": quantity, "total": order.total, "status": "confirmed",
        }

    @is_tool(ToolType.WRITE)
    def modify_order(
        self, order_id: str,
        field: Literal["quantity", "shipping_address"],
        new_value: str,
    ) -> dict:
        """Modify an existing order before fulfillment is finalized.

        Only fields in _MODIFIABLE_FIELDS["modify_order"] may be changed;
        status / total / ids / payment_method go through their own flows.
        """
        allowed = self._MODIFIABLE_FIELDS.get("modify_order", set())
        if field not in allowed:
            raise ValueError(
                f"Field '{field}' is not modifiable on an order. "
                f"Allowed: {sorted(allowed)}"
            )
        o = self.db.orders.get(order_id)
        if not o:
            raise ValueError(f"Order not found: {order_id}")
        if field == "quantity":
            new_value = parse_int(new_value, "new_value")
        setattr(o, field, new_value)
        return {"order_id": order_id, "status": "modified", "field": field}

    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: str) -> dict:
        """Cancel an existing order."""
        o = self.db.orders.get(order_id)
        if not o:
            raise ValueError(f"Order not found: {order_id}")
        o.status = "cancelled"
        return {"order_id": order_id, "status": "cancelled"}

    @is_tool(ToolType.READ)
    def track_order(
        self,
        order_id: str,
        update_scope: Optional[Literal["critical", "all", "schedule_change"]] = None,
    ) -> dict:
        """Check shipping or fulfillment status for an order. update_scope
        narrows what the caller wants to monitor (no-op on the stub since
        ATR only tests that the call shape matches gold; real backends
        would filter by scope).
        """
        o = self.db.orders.get(order_id)
        if not o:
            raise ValueError(f"Order not found: {order_id}")
        if update_scope is not None:
            parse_enum(update_scope, {"critical", "all", "schedule_change"}, "update_scope")
        return {"order_id": order_id, "status": o.status, "update_scope": update_scope}

    @is_tool(ToolType.WRITE)
    def return_order(
        self, order_id: str,
        resolution_type: Optional[ResolutionType] = None,
    ) -> dict:
        """Initiate a return or exchange workflow for an existing order.
        resolution_type is encoded into order.status so downstream calls
        (e.g. track_order) observe a real difference by branch —
        'return_requested_refund' vs 'return_requested_exchange' vs
        'return_requested_store_credit' vs 'return_requested'.
        """
        o = self.db.orders.get(order_id)
        if not o:
            raise ValueError(f"Order not found: {order_id}")
        if resolution_type is not None:
            parse_enum(resolution_type, {"refund", "exchange", "store_credit"}, "resolution_type")
        o.status = (
            f"return_requested_{resolution_type}"
            if resolution_type
            else "return_requested"
        )
        return {"order_id": order_id, "status": o.status,
                "resolution": resolution_type}

    @is_tool(ToolType.READ)
    def review_recurring_charges(
        self,
        focus: Literal["price_change", "usage", "tier", "all"] = "all",
    ) -> dict:
        """Inspect recurring charges, price changes, or unusual cost updates.

        The `focus` filter narrows results by the anomaly signal carried in
        each subscription's `attributes`:
          - price_change → subs flagged with `price_changed` or carrying a
            different `previous_price_per_period`
          - usage        → subs flagged with `usage_anomaly` or `usage_notes`
          - tier         → subs flagged with `plan_changed` or `tier_change`
          - all          → every subscription, unfiltered

        Data construction places these flags in reference attributes so the
        focus parameter produces a genuinely different result set.
        """
        parse_enum(focus, {"price_change", "usage", "tier", "all"}, "focus")
        subs = list(self.db.subscriptions.values())
        if focus == "price_change":
            subs = [s for s in subs if _has_attr(s, "price_changed")
                    or _has_attr(s, "previous_price_per_period")]
        elif focus == "usage":
            subs = [s for s in subs if _has_attr(s, "usage_anomaly")
                    or _has_attr(s, "usage_notes")]
        elif focus == "tier":
            subs = [s for s in subs if _has_attr(s, "plan_changed")
                    or _has_attr(s, "tier_change")]
        # focus == "all": no filter
        return {
            "focus": focus,
            "count": len(subs),
            "subscriptions": [s.model_dump() for s in subs],
        }

    @is_tool(ToolType.WRITE)
    def pause_subscription(
        self, subscription_id: str,
        pause_until: Optional[str] = None,
    ) -> dict:
        """Pause a recurring subscription or service."""
        s = self.db.subscriptions.get(subscription_id)
        if not s:
            raise ValueError(f"Subscription not found: {subscription_id}")
        if pause_until is not None:
            pause_until = parse_iso_date(pause_until, "pause_until")
        s.status = "paused"
        s.pause_until = pause_until
        return {"subscription_id": subscription_id, "status": "paused"}

    @is_tool(ToolType.WRITE)
    def resume_subscription(
        self, subscription_id: str,
    ) -> dict:
        """Resume a paused subscription or service."""
        s = self.db.subscriptions.get(subscription_id)
        if not s:
            raise ValueError(f"Subscription not found: {subscription_id}")
        s.status = "active"
        s.pause_until = None
        return {"subscription_id": subscription_id, "status": "active"}

    @is_tool(ToolType.WRITE)
    def change_subscription_plan(
        self, subscription_id: str, new_plan: SubscriptionPlan,
    ) -> dict:
        """Change the plan or tier of a subscription.

        Args:
            subscription_id: The subscription to modify.
            new_plan: Target tier (basic | pro | premium | enterprise).
        """
        parse_enum(new_plan, {"basic", "pro", "premium", "enterprise"}, "new_plan")
        s = self.db.subscriptions.get(subscription_id)
        if not s:
            raise ValueError(f"Subscription not found: {subscription_id}")
        s.plan = new_plan
        return {"subscription_id": subscription_id, "plan": new_plan}

    @is_tool(ToolType.WRITE)
    def cancel_subscription(self, subscription_id: str) -> dict:
        """Cancel a recurring subscription or service."""
        s = self.db.subscriptions.get(subscription_id)
        if not s:
            raise ValueError(f"Subscription not found: {subscription_id}")
        s.status = "cancelled"
        return {"subscription_id": subscription_id, "status": "cancelled"}


def _has_attr(sub: Subscription, key: str) -> bool:
    """Truthy presence check on a subscription attribute."""
    return bool(sub.attributes.get(key))


# ---------------------------------------------------------------------------
# Env builder (called from domains/__init__.py registry)
# ---------------------------------------------------------------------------

def build_env(
    session_task,
    persona: PersonaProfile,
    allowed_tools: list[str],
    seed: Optional[int] = None,
) -> ATREnv:
    """Build a session-scoped ATREnv for the commerce domain."""
    db = CommerceDB.from_references(persona, session_task.local_env.references)
    toolkit = CommerceTools(db)
    return ATREnv(
        domain="commerce",
        toolkit=toolkit,
        allowed_tools=allowed_tools,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TableSpec declarations + registry
# ---------------------------------------------------------------------------


def _build_product_row(ref_id, attrs, persona, spec):
    return {
        "product_id": ref_id,
        "name": attrs.get("name", ref_id),
        "price": float(attrs.get("price", 0)),
        "category": attrs.get("category", "unknown"),
        "rating": attrs.get("rating"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "price", "category", "rating")},
    }


def _build_subscription_row(ref_id, attrs, persona, spec):
    return {
        "subscription_id": ref_id,
        "name": attrs.get("name", ref_id),
        "plan": attrs.get("plan", "standard"),
        "status": attrs.get("status", "active"),
        "pause_until": attrs.get("pause_until"),
        "price_per_period": attrs.get("price_per_period"),
        "attributes": {k: v for k, v in attrs.items()
                       if k not in ("name", "plan", "status",
                                    "pause_until", "price_per_period")},
    }


def _derive_order_row(source_ref_id, source_attrs, persona, spec):
    return {
        "order_id": source_attrs["order_id"],
        "product_id": source_ref_id,
        "quantity": 1,
        "shipping_address": persona.default_shipping_address,
        "payment_method": persona.default_payment_method,
        "total": float(source_attrs.get("price", 0)),
        "status": (source_attrs.get("order_status")
                   or source_attrs.get("delivery_status", "confirmed")),
    }


_COMMERCE_TABLES = [
    TableSpec(
        name="products", model=Product, kind="primary",
        source_ref_type="product",
        promoted_attrs=["name", "price", "category", "rating"],
        operating_tools=["search_products", "compare_products",
                         "place_order", "build_shopping_list"],
        discovery_tools=["search_products"],
        build_row=_build_product_row,
    ),
    TableSpec(
        name="subscriptions", model=Subscription, kind="primary",
        source_ref_type="subscription",
        promoted_attrs=["name", "plan", "status",
                        "pause_until", "price_per_period"],
        operating_tools=["review_recurring_charges",
                         "pause_subscription", "resume_subscription",
                         "change_subscription_plan", "cancel_subscription"],
        discovery_tools=["review_recurring_charges"],
        build_row=_build_subscription_row,
    ),
    TableSpec(
        name="orders", model=Order, kind="derived",
        source_ref_type="product", source_attr="order_id",
        operating_tools=["modify_order", "cancel_order",
                         "return_order", "track_order"],
        discovery_tools=["search_products"],
        derive_row=_derive_order_row,
    ),
]

CommerceDB._TABLES = _COMMERCE_TABLES
register_domain_tables("commerce", _COMMERCE_TABLES)
