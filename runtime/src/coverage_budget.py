"""Request-local admission and single-use provenance for a cold access plan."""
from dataclasses import dataclass


class VisitBudget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError('visit budget must be a nonnegative integer')
        self.limit = limit
        self._consumed = 0

    @property
    def consumed(self):
        return self._consumed

    def reserve(self, raw_visits):
        if type(raw_visits) is not int or raw_visits < 0:
            raise ValueError('raw visits must be a nonnegative integer')
        if self._consumed + raw_visits > self.limit:
            return False
        self._consumed += raw_visits
        return True


@dataclass(frozen=True)
class Decision:
    request: int
    opinion: str
    parent: str
    epoch: int
    target: int


@dataclass(frozen=True, eq=False)
class PlanTicket:
    decision: Decision
    anchor: object
    index: object
    probe: object
    drift: object
    excluded: set


class PlanTickets:
    """Issue one plan from trusted request-local inputs and retire it before use.

    The cursor exclusively owns the drift and exclusions until gather. Their
    identities travel with the ticket, without copying or hashing the payloads.
    """
    def __init__(self):
        self._pending = None

    def issue(self, decision, anchor, probe, drift, excluded):
        if self._pending is not None:
            raise ValueError('previous access plan is still active')
        ticket = PlanTicket(decision, anchor, anchor.index, probe, drift, excluded)
        self._pending = ticket
        return ticket

    def take(self, ticket, decision, anchor):
        if (self._pending is None or ticket is not self._pending or ticket.decision is not decision or
                ticket.anchor is not anchor or ticket.index is not anchor.index):
            raise ValueError('plan binding mismatch')
        self._pending = None
        return ticket
