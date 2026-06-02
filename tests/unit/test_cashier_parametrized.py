"""
tests/unit/test_cashier_parametrized.py
----------------------------------------
S-8: Parametrized coverage of all 14 cashier state-machine rules (A1–A7, N1–N6).

Each row: (description, cz_kwargs, kz_kwargs, now, unauthorized, expected_case_ids)

expected_case_ids is a set so tests remain resilient to minor priority re-ordering
at the same severity level.
"""

from __future__ import annotations

import pytest

from services.cashier import (
    SEVERITY_ALERT,
    SEVERITY_CRITICAL,
    SEVERITY_NORMAL,
    CashierService,
    ZoneCount,
)


def _svc(wait_max: float = 30.0, drawer_max: float = 60.0) -> CashierService:
    """Build a minimal CashierService with no-op model so we can call _evaluate directly."""
    svc = object.__new__(CashierService)
    svc._wait_max = wait_max
    svc._drawer_max = drawer_max
    svc._enable_staff_list = False
    svc._nearby = lambda pa, db: bool(pa and db)
    svc._all_nearby = lambda pa, db: bool(pa and db)
    svc._drawer_open_since = None
    svc._customer_wait_since = None
    return svc


def _zc(persons=0, drawers=0, cash=0, person_bboxes=None, drawer_bboxes=None) -> ZoneCount:
    return ZoneCount(
        persons=persons,
        drawers=drawers,
        cash=cash,
        person_bboxes=person_bboxes or [],
        drawer_bboxes=drawer_bboxes or [],
    )


# ── Parametrized table ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "label,cz,kz,now,unauthorized,expected_case,expected_sev",
    [
        # A3: cash + open drawer + no cashier
        (
            "A3 open drawer+cash no cashier",
            _zc(persons=0, drawers=1, cash=1),
            _zc(),
            1.0, False, "A3", SEVERITY_CRITICAL,
        ),
        # A1 CRITICAL: open drawer + customer in KZ + no cashier
        (
            "A1 CRITICAL unattended + customer",
            _zc(persons=0, drawers=1, cash=0),
            _zc(persons=1),
            1.0, False, "A1", SEVERITY_CRITICAL,
        ),
        # A1 ALERT: open drawer, no cash, no cashier, no customer
        (
            "A1 ALERT unattended drawer",
            _zc(persons=0, drawers=1, cash=0),
            _zc(persons=0),
            1.0, False, "A1", SEVERITY_ALERT,
        ),
        # A2: unauthorized in CZ
        (
            "A2 unauthorized person",
            _zc(persons=1),
            _zc(),
            1.0, True, "A2", SEVERITY_ALERT,
        ),
        # A7: cash in KZ, no cashier
        (
            "A7 cash in customer zone",
            _zc(persons=0, drawers=0),
            _zc(cash=1),
            1.0, False, "A7", SEVERITY_ALERT,
        ),
        # A5 tested separately via test_a5_customer_wait_over_threshold (needs 2 calls to prime timer)

        # N3: transaction in progress (cashier + customer + drawer + cash)
        (
            "N3 transaction in progress",
            _zc(persons=1, drawers=1, cash=1, person_bboxes=[[0,0,10,10]], drawer_bboxes=[[0,0,10,10]]),
            _zc(persons=1),
            1.0, False, "N3", SEVERITY_NORMAL,
        ),
        # N4: two staff + drawer open + all near drawer
        (
            "N4 handover",
            _zc(persons=2, drawers=1, person_bboxes=[[0,0,1,1],[2,2,3,3]], drawer_bboxes=[[0,0,3,3]]),
            _zc(),
            1.0, False, "N4", SEVERITY_NORMAL,
        ),
        # N6: cashier + drawer open, no cash
        (
            "N6 drawer open no cash",
            _zc(persons=1, drawers=1, cash=0),
            _zc(cash=0),
            1.0, False, "N6", SEVERITY_NORMAL,
        ),
        # N2: cashier present, drawer closed
        (
            "N2 cashier present idle",
            _zc(persons=1, drawers=0),
            _zc(),
            1.0, False, "N2", SEVERITY_NORMAL,
        ),
        # N5: customer waiting within limit
        (
            "N5 customer waiting within limit",
            _zc(persons=0, drawers=0, cash=0),
            _zc(persons=1),
            5.0, False, "N5", SEVERITY_NORMAL,
        ),
        # N1: idle
        (
            "N1 idle",
            _zc(),
            _zc(),
            1.0, False, "N1", SEVERITY_NORMAL,
        ),
    ],
)
def test_evaluate_case(label, cz, kz, now, unauthorized, expected_case, expected_sev):
    svc = _svc()
    case_id, severity, alerts, txn, drawer_ms, wait_ms = svc._evaluate(cz, kz, now, unauthorized)
    assert case_id == expected_case, f"[{label}] expected {expected_case}, got {case_id}"
    assert severity == expected_sev, f"[{label}] expected severity {expected_sev}, got {severity}"


def test_a5_customer_wait_over_threshold():
    """A5 requires two calls: first primes the timer, second fires when elapsed > max."""
    svc = _svc(wait_max=10.0)
    cz = _zc(persons=0, drawers=0, cash=0)
    kz = _zc(persons=1)
    # prime timer at t=0
    svc._evaluate(cz, kz, 0.0, False)
    case_id, severity, alerts, *_ = svc._evaluate(cz, kz, 11.0, False)
    assert case_id == "A5"
    assert severity == SEVERITY_ALERT


def test_a6_drawer_overtime():
    """A6: cashier present + nearby drawer open too long (no cash to avoid A4)."""
    svc = _svc(drawer_max=5.0)
    # No cash in CZ to avoid A4 legacy path; cashier IS near drawer
    cz = _zc(persons=1, drawers=1, cash=0, person_bboxes=[[0,0,10,10]], drawer_bboxes=[[0,0,10,10]])
    kz = _zc()
    # prime the drawer timer at t=0
    svc._evaluate(cz, kz, 0.0, False)
    case_id, severity, alerts, *_ = svc._evaluate(cz, kz, 6.0, False)
    assert case_id == "A6"
    assert severity == SEVERITY_ALERT
    assert "6" in alerts[0] or "A6" in alerts[0]


def test_a4_legacy_no_staff_list_cashier_far_from_drawer():
    """A4 legacy path: cash present, cashier NOT near drawer → unauthorized."""
    svc = _svc()
    svc._enable_staff_list = False
    svc._nearby = lambda pa, db: False  # force no proximity

    cz = _zc(persons=1, drawers=1, cash=1, person_bboxes=[[0,0,1,1]], drawer_bboxes=[[100,100,200,200]])
    case_id, severity, *_ = svc._evaluate(cz, _zc(), 1.0, False)
    assert case_id == "A4"
    assert severity == SEVERITY_CRITICAL


def test_n3_is_a_transaction():
    """N3 must set the transaction flag."""
    svc = _svc()
    cz = _zc(persons=1, drawers=1, cash=1, person_bboxes=[[0,0,10,10]], drawer_bboxes=[[0,0,10,10]])
    kz = _zc(persons=1)
    *_, txn, _dm, _wm = svc._evaluate(cz, kz, 1.0, False)
    assert txn is True
