"""One-hop demand planner. Cost predictions use deterministic work proxies."""
from dataclasses import dataclass
from time import perf_counter_ns

from policy_kernel import PolicyKernel


def gate_costs(candidate_rank, winner_rank, rhs_units, parent_rhs_units,
               has_response, eligible_requests, forecast=0, replay_work=0):
    """Use rank one for forecast fresh work and rank two for its transfer."""
    if has_response:
        forecast = 0
    fresh = (candidate_rank + forecast) * rhs_units
    transport = 128 * (candidate_rank + 2*forecast) * winner_rank
    if not has_response:
        transport += winner_rank * parent_rhs_units + replay_work
    return (has_response or eligible_requests >= 2) and transport < fresh, fresh, transport


@dataclass
class Pending:
    winner: int
    rows: dict
    rank: int
    rhs_units: int
    operator: tuple | None
    response: tuple | None
    replay_work: int = 0
    pivot: tuple | None = None
    requests: int = 0
    forecast_left: int = 0


class CostPlanner(PolicyKernel):
    """Single mutable path, with at most one generation of pending Grams."""
    def __init__(self, *args, winner_slots=1, **kwargs):
        super().__init__(*args, winner_slots=winner_slots, **kwargs)
        if not self.cached or self.cached == "same":
            raise ValueError("the planner requires a current-version Gram cache")
        self.pending = None
        self.last_iterations = 1
        self._factor_entries = None

    def solve(self, rhs):
        before = self.ledger["cg_iterations"]
        result = super().solve(rhs)
        self.last_iterations = max(1, self.ledger["cg_iterations"] - before)
        return result

    def _rhs_units(self):
        # Two sparse products for the forward residual and its magnitude bound.
        n, nz = self.A.shape[0], self.A.nnz
        if self.factor is None:
            return (2*nz+10*n)*self.last_iterations + 4*nz
        if self._factor_entries is None:
            self._factor_entries = self.factor.stored_entries()
        return 4*(self._factor_entries+nz)

    def _dispose_parent(self):
        if self.pending is not None and self.pending.operator is not None:
            self.pending.operator = None

    def _operator(self):
        return self.A, self.absA, self.gamma, self.prec, self.invdiag, self.factor

    def _bind_operator(self, operator):
        self.A, self.absA, self.gamma, self.prec, self.invdiag, self.factor = operator

    def commit(self, e):
        if not 0 <= e < len(self.edits) or self.active[e]:
            raise ValueError("invalid commitment")
        self._dispose_parent()
        self.pending = None
        self.rows.pop(e, None)
        if self.rows:
            response = self._winner_columns.get(e) if self._winner_operator is self.A else None
            rank = 1 + int(self.flip[e])
            units = self._rhs_units()
            operator = None
            replay_work = 0
            if response is None and len(self.rows) >= 2:
                # Parent arrays are immutable. A direct factor is reconstructed
                # only after the child provides evidence warranting its cost.
                operator = (*self._operator()[:-1], None)
                if self.factor is not None:
                    replay_work = 2*(1+rank)*self._factor_entries
            self.pending = Pending(e, self.rows, rank, units, operator, response, replay_work)
        self.rows, self.stale = {}, set()
        super().commit(e)
        self._factor_entries = None

    def _historical(self, t):
        start = perf_counter_ns()
        child = self._operator()
        iterations = self.last_iterations
        clone = None
        try:
            if self.factor is not None:
                clone = self.factor.clone()
                self.ledger["factor_clones"] += 1
                self.ledger["planner_parent_clones"] += 1
                clone.undo(self.edits[t.winner])
                self.ledger["planner_undo_down"] += 1
                self.ledger["planner_undo_up"] += int(self.flip[t.winner])
                self.ledger["peak_planner_clone_bytes"] = max(
                    self.ledger["peak_planner_clone_bytes"], clone.peak_bytes)
            self._bind_operator((*t.operator[:-1], clone))
            return self._fresh(t.winner, "W")
        finally:
            self._bind_operator(child)
            self.last_iterations = iterations
            if clone is not None:
                clone.close()
            self.ledger["historical_ns"] += perf_counter_ns()-start

    def _interval(self, e, z, rho, refresh=False, capture=None):
        t = self.pending
        if not refresh and t is not None and e not in self.rows and e in t.rows:
            t.requests += 1
            self.ledger["gate_requests"] += 1
            if t.forecast_left:
                t.forecast_left -= 1
                self.ledger["forecast_still_unrealized"] -= 1
                self.ledger["forecast_realized"] += 1
            rank = 1 + int(self.flip[e])
            # Repeat the observed prefix length less one, capped by available
            # old rows. Previously paid requests contribute no claimed saving.
            forecast = min(t.requests-1, len(t.rows)-1)
            # Direct solves use the same parent-version proxy on both sides.
            child_units = self._rhs_units() if self.factor is None else t.rhs_units
            opened, fresh, transport = gate_costs(
                rank, t.rank, child_units, t.rhs_units, t.response is not None, t.requests,
                forecast, t.replay_work)
            self.ledger["predicted_fresh_units"] += fresh
            self.ledger["predicted_transport_units"] += transport
            if opened and (t.response is not None or t.operator is not None):
                self.ledger["gate_opens"] += 1
                if t.pivot is None:
                    if t.response is None:
                        t.response = self._historical(t)
                        t.forecast_left = forecast
                        self.ledger["forecast_rows"] += forecast
                        self.ledger["forecast_still_unrealized"] += forecast
                        self._dispose_parent()
                    else:
                        self.ledger["retained_hits"] += 1
                        self.ledger["winner_column_hits"] += 1
                    _, G, err = t.response
                    t.pivot = self._pivot(t.winner, G, err)
                self._materialize(e,t)
                self.ledger["transported"] += 1
            else:
                self.ledger["gate_misses"] += 1
            del t.rows[e]
            if not t.rows:
                self._dispose_parent()
                self.pending = None
        return super()._interval(e, z, rho, refresh=refresh, capture=capture)

    def _materialize(self,e,t):
        current = self.rows
        self.rows = {e:t.rows[e]}
        try:
            Y, _, err = t.response
            self._transport(Y, *t.pivot, err)
            current.update(self.rows)
        finally:
            self.rows = current

    def memory_bytes(self):
        size = super().memory_bytes()
        t = self.pending
        if t is None:
            return size
        size += sum(row[0].nbytes+8 for row in t.rows.values())
        if t.response is not None:
            size += t.response[0].nbytes
        if t.operator is not None:
            A, absA, gamma, _, invdiag, factor = t.operator
            size += gamma.nbytes + invdiag.nbytes + (factor.bytes if factor else 0)
            for matrix in (A,absA):
                size += matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        return size

    def close(self):
        self._dispose_parent()
        self.pending = None
        super().close()
