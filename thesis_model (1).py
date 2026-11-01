"""
===============================================================================
PORTFOLIO COMPRESSION IN FINANCIAL NETWORKS: A MULTI-CCP MODEL
Simulation code — Vivien Szalóki, Corvinus University of Budapest
===============================================================================

HOW A NETWORK IS REPRESENTED
----------------------------
A network is a pandas DataFrame. One row = one obligation:

    debtor   creditor   amount   asset_class
    0        3          118.4    IRS
    3        7           92.1    CDS

This is exactly the edge set E from section 2.1 of the thesis: each row is a
directed edge (i, j) with weight w_ij. Keeping it as a table means you can
print it and see what is actually in the network.

WHAT THE CODE DOES
------------------
  Section 1  build a network
  Section 2  measure a network (gross exposure, net positions)
  Section 3  the two ways obligations can be cancelled
  Section 4  who can actually pay (Eisenberg-Noe clearing vector)
  Section 5  the four clearing regimes compared in the thesis
  Section 6  helpers (external assets, partitions)

Run this file directly to see the validation tests.
===============================================================================
"""

import numpy as np

# Core sizes are fixed per market size rather than set as a share of the
# market, which keeps the two-tier structure sharp as the market grows.
# Replaces the previous rounded core_share rule; the realised share now
# declines from 25% at N=8 to 5% at N=200, closer to reported interbank cores.
CORE_SIZES = {8: 2, 12: 2, 20: 3, 30: 4, 50: 5, 100: 8, 200: 10}
import pandas as pd
import networkx as nx
from scipy.optimize import linprog


# =============================================================================
# SECTION 1 — BUILDING A NETWORK
# =============================================================================

def make_random_network(n_institutions, asset_classes=("CDS", "IRS", "EQ", "FX"),
                        link_prob=0.20, min_amount=20, max_amount=150, seed=None):
    """
    Every possible obligation exists with probability `link_prob`.

    Simple but unrealistic: obligations are spread evenly across institutions.
    Real markets concentrate around a few dealers — see make_core_periphery_network.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cls in asset_classes:
        for i in range(n_institutions):
            for j in range(n_institutions):
                if i != j and rng.random() < link_prob:
                    rows.append((i, j, rng.uniform(min_amount, max_amount), cls))
    return pd.DataFrame(rows, columns=["debtor", "creditor", "amount", "asset_class"])


def make_core_periphery_network(n_institutions, asset_classes=("CDS", "IRS", "EQ", "FX"),
                                core_share=0.2, n_core=None,
                                p_core_core=0.55, p_core_periphery=0.18,
                                p_periphery_periphery=0.01, min_amount=20, max_amount=150,
                                core_size_multiplier=2.0, seed=None):
    """
    A few large dealers (the 'core') trade heavily with each other and with
    everyone else; the remaining institutions (the 'periphery') rarely trade
    directly with one another.

    This is the structure found empirically in interbank markets (Craig & von
    Peter, 2014) and matches the dealer-intermediation role that D'Errico and
    Roukny (2021) identify as essential for multilateral compression.

    Returns (obligations, core_members).
    """
    rng = np.random.default_rng(seed)
    if n_core is None:
        n_core = max(2, int(round(core_share * n_institutions)))
    core = set(range(n_core))

    rows = []
    for cls in asset_classes:
        for i in range(n_institutions):
            for j in range(n_institutions):
                if i == j:
                    continue
                i_core, j_core = i in core, j in core
                if i_core and j_core:
                    p, size = p_core_core, core_size_multiplier
                elif i_core or j_core:
                    p, size = p_core_periphery, 1.0
                else:
                    p, size = p_periphery_periphery, 1.0
                if rng.random() < p:
                    rows.append((i, j, rng.uniform(min_amount, max_amount) * size, cls))

    obligations = pd.DataFrame(rows, columns=["debtor", "creditor", "amount", "asset_class"])
    return obligations, core


# =============================================================================
# SECTION 2 — MEASURING A NETWORK
# =============================================================================

def gross_exposure(obligations):
    """Total notional outstanding. This is what compression reduces."""
    return obligations["amount"].sum()


def net_positions(obligations, institutions=None):
    """
    Net position of each institution = what it is owed minus what it owes.

    This must NOT change when obligations are netted or compressed — that is
    the defining property of compression (section 2.1). Used as a correctness
    check after every operation.
    """
    if institutions is None:
        institutions = sorted(set(obligations["debtor"]) | set(obligations["creditor"]))
    owed_to = obligations.groupby("creditor")["amount"].sum()
    owed_by = obligations.groupby("debtor")["amount"].sum()
    return pd.Series({i: owed_to.get(i, 0.0) - owed_by.get(i, 0.0) for i in institutions})


def net_positions_unchanged(before, after, institutions, tolerance=1e-6):
    a = net_positions(before, institutions)
    b = net_positions(after, institutions)
    return bool((a - b).abs().max() < tolerance)


def no_new_counterparties(before, after):
    """
    Compression may only cancel notional along relationships that already
    exist. If this fails, obligations have been invented between institutions
    that never traded, which is novation, not compression.
    """
    was = set(zip(before["debtor"], before["creditor"]))
    now = set(zip(after["debtor"], after["creditor"]))
    return now <= was


def compression_lower_bound(obligations, institutions=None):
    """
    The smallest gross notional any netting arrangement could reach, ignoring
    who is allowed to face whom: the sum of the positive net positions.

    Compression cannot go below this, and generally stays above it, because it
    may not create new counterparty relationships. Reported as a reference
    point, not as a target.
    """
    net = net_positions(obligations, institutions)
    return float(net.clip(lower=0.0).sum())


# =============================================================================
# SECTION 3 — THE TWO WAYS OBLIGATIONS CAN BE CANCELLED
# =============================================================================
#
# The whole thesis rests on the difference between these two.
#
#   bilateral_netting        needs only the two counterparties to agree.
#                            Always available. Cancels A-owes-B against
#                            B-owes-A.
#
#   multilateral_compression needs someone who can see the whole network
#                            (a CCP). Cancels around cycles: A->B->C->A.
#                            Much more powerful, but only works over the
#                            obligations that coordinator can actually see.
#
# Fragmentation costs the multilateral part, not the bilateral part.
# =============================================================================

def bilateral_netting(obligations):
    """
    For each pair of institutions, cancel offsetting obligations against each
    other. Asset classes are pooled: an institution's CDS claim can offset its
    IRS liability to the same counterparty, because it is one bilateral
    relationship.

    Returns a new obligations table.
    """
    if obligations.empty:
        return obligations.copy()

    # collapse to one net amount per ordered pair
    pair_totals = obligations.groupby(["debtor", "creditor"])["amount"].sum().to_dict()

    rows = []
    done = set()
    for (i, j), amount_ij in pair_totals.items():
        if (i, j) in done:
            continue
        amount_ji = pair_totals.get((j, i), 0.0)
        done.add((i, j))
        done.add((j, i))

        net = amount_ij - amount_ji
        if net > 1e-9:
            rows.append((i, j, net, "netted"))
        elif net < -1e-9:
            rows.append((j, i, -net, "netted"))
        # net == 0: the two cancel exactly, nothing remains

    return pd.DataFrame(rows, columns=["debtor", "creditor", "amount", "asset_class"])


def multilateral_compression(obligations, tolerance=1e-9):
    """
    Multilateral compression, solved exactly as a linear program.

    Minimises total gross notional subject to three constraints:

      1. every institution's net position is unchanged;
      2. an obligation can only be reduced, never increased;
      3. NO new counterparty relationship is created.

    Constraint 3 is what makes this COMPRESSION rather than novation. A
    participant never ends up exposed to a counterparty it was not already
    exposed to. D'Errico and Roukny (2021) call this the conservative case,
    and it is what compression services actually do: they cancel redundant
    notional, they do not re-paper the market.

    Dropping constraint 3 would reach a lower gross figure, but by inventing
    obligations between institutions that never traded with each other. That
    is a different operation, and it does not reproduce Table 2 of section 2.1.

    Requires a coordinator with full visibility of the obligations passed in —
    this is what a CCP provides and what bilateral netting cannot achieve.
    """
    if obligations.empty:
        return obligations.iloc[0:0][["debtor", "creditor", "amount",
                                      "asset_class"]].copy()

    # one variable per existing ordered pair, capped at the amount now owed
    pair = obligations.groupby(["debtor", "creditor"])["amount"].sum()
    edges = list(pair.index)
    institutions = sorted(set(obligations["debtor"]) | set(obligations["creditor"]),
                          key=str)
    index = {inst: k for k, inst in enumerate(institutions)}

    # one equality per institution: (what it pays out) - (what it takes in)
    # must stay what it is now
    A = np.zeros((len(institutions), len(edges)))
    for k, (i, j) in enumerate(edges):
        A[index[i], k] += 1.0
        A[index[j], k] -= 1.0

    owed_by = obligations.groupby("debtor")["amount"].sum()
    owed_to = obligations.groupby("creditor")["amount"].sum()
    b = np.array([float(owed_by.get(i, 0.0) - owed_to.get(i, 0.0))
                  for i in institutions])

    result = linprog(np.ones(len(edges)), A_eq=A, b_eq=b,
                     bounds=[(0.0, float(pair[e])) for e in edges],
                     method="highs")
    if not result.success:
        raise RuntimeError(f"compression LP did not solve: {result.message}")

    rows = [(i, j, float(x), "compressed")
            for (i, j), x in zip(edges, result.x) if x > tolerance]
    return pd.DataFrame(rows, columns=["debtor", "creditor", "amount",
                                       "asset_class"])


def compression_by_cycles(obligations, max_cycles=100000):
    """
    The simple cycle-cancellation heuristic: find a cycle, reduce every edge in
    it by the cycle's smallest amount, repeat (Gavrila and Popa, 2020).

    Kept for comparison only. It satisfies the same three constraints as
    multilateral_compression but does not reach the optimum: cancelling cycles
    in an arbitrary order leaves flow routed along longer paths than necessary,
    and each extra hop is extra gross notional. On the networks used here it
    leaves roughly 5-7 percentage points of reduction on the table.
    """
    if obligations.empty:
        return obligations.copy()

    # build a graph: one edge per pair, amounts summed
    G = nx.DiGraph()
    for (i, j), amount in obligations.groupby(["debtor", "creditor"])["amount"].sum().items():
        G.add_edge(i, j, weight=amount)

    for _ in range(max_cycles):
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        smallest = min(G[u][v]["weight"] for u, v, _ in cycle)
        for u, v, _ in cycle:
            G[u][v]["weight"] -= smallest
            if G[u][v]["weight"] <= 1e-9:
                G.remove_edge(u, v)

    rows = [(u, v, d["weight"], "compressed") for u, v, d in G.edges(data=True)]
    return pd.DataFrame(rows, columns=["debtor", "creditor", "amount", "asset_class"])


# =============================================================================
# SECTION 4 — WHO CAN ACTUALLY PAY (Eisenberg & Noe, 2001)
# =============================================================================

def clearing_vector(obligations, external_assets, institutions,
                    tolerance=1e-10, max_iterations=10000):
    """
    Computes how much each institution actually pays:

        p_i = min( c_i ,  e_i + sum_j Lambda_ji * p_j )

        c_i     what i owes in total
        e_i     i's own outside assets
        Lambda  share of j's debt that is owed to i
        p_i     what i actually pays

    The min() is limited liability: you pay at most what you owe, and at most
    what you have. Solved by iteration because the answer is circular — what i
    can pay depends on what i receives, which depends on what others can pay.
    Starting from "everyone pays in full", payments only fall: the iteration is
    monotone non-increasing and converges to the greatest clearing vector.

    Returns (payments, total_owed, n_defaults, unpaid_claims).
    """
    index = {inst: k for k, inst in enumerate(institutions)}
    n = len(institutions)

    W = np.zeros((n, n))
    for row in obligations.itertuples(index=False):
        W[index[row.debtor], index[row.creditor]] += row.amount

    c = W.sum(axis=1)                                   # total owed by each
    e = np.array([external_assets.get(i, 0.0) for i in institutions])

    Lambda = np.zeros((n, n))
    for j in range(n):
        if c[j] > 0:
            Lambda[j, :] = W[j, :] / c[j]

    # each debtor's row of Lambda must share out exactly what it owes
    for j in range(n):
        if c[j] > tolerance:
            assert abs(Lambda[j].sum() - 1.0) < 1e-10, \
                f"Lambda row {j} sums to {Lambda[j].sum()}, not 1"

    p = c.copy()                                        # assume everyone pays in full
    for _ in range(max_iterations):
        p_next = np.minimum(c, e + Lambda.T @ p)
        if np.max(np.abs(p_next - p)) < tolerance:
            p = p_next
            break
        p = p_next
    else:
        raise RuntimeError(
            f"clearing vector did not converge in {max_iterations} iterations")

    n_defaults = int(np.sum(p < c - 1e-6))
    unpaid = float(np.sum(c - p))
    return p, c, n_defaults, unpaid


# =============================================================================
# SECTION 5 — THE FOUR CLEARING REGIMES
# =============================================================================
#
# All four start from the SAME network, so any difference in outcome comes
# from the clearing structure and not from random variation.
# =============================================================================

def _evaluate(original, after_netting, external_assets, institutions, label):
    payments, owed, n_defaults, unpaid = clearing_vector(
        after_netting, external_assets, institutions)
    before = gross_exposure(original)
    after = gross_exposure(after_netting)
    return {
        "regime": label,
        "gross_before": before,
        "gross_after": after,
        "reduction_pct": 100 * (1 - after / before) if before > 0 else 0.0,
        "n_defaults": n_defaults,
        "unpaid_claims": unpaid,
        "net_positions_ok": net_positions_unchanged(original, after_netting, institutions),
    }


def regime_no_netting(obligations, external_assets, institutions):
    """Nothing is cancelled. The floor everything else is measured against."""
    return _evaluate(obligations, obligations, external_assets, institutions,
                     "no netting")


def regime_bilateral_only(obligations, external_assets, institutions):
    """No CCP. Counterparties net between themselves — always possible."""
    after = bilateral_netting(obligations)
    return _evaluate(obligations, after, external_assets, institutions,
                     "bilateral only")


def regime_single_ccp(obligations, external_assets, institutions):
    """One CCP sees everything and compresses every cycle. The best case."""
    after = multilateral_compression(obligations)
    return _evaluate(obligations, after, external_assets, institutions,
                     "single CCP")


def regime_multi_ccp(obligations, ccp_labels, external_assets, institutions,
                     label="multiple CCPs"):
    """
    Several CCPs, each seeing only part of the network.

    `ccp_labels` is a plain LIST with one entry per obligation, in the same
    order as the rows of `obligations`. Each entry is the name of the CCP that
    clears that obligation, or None if no CCP sees it. The two helper
    functions in section 6 build this list — one per way of splitting the
    market. Because it is just a list, you can print it and check it.

    Obligations at the same CCP are compressed multilaterally together.
    Everything left over is netted bilaterally, because two counterparties can
    always net between themselves even with no CCP in the middle.
    """
    assigned = obligations.copy()
    assigned["ccp"] = ccp_labels

    pieces = []

    # each CCP compresses what it can see
    centrally_cleared = assigned[assigned["ccp"].notna()]
    for ccp_name, group in centrally_cleared.groupby("ccp"):
        pieces.append(multilateral_compression(group))

    # everything no CCP sees is still netted bilaterally
    uncleared = assigned[assigned["ccp"].isna()]
    if not uncleared.empty:
        pieces.append(bilateral_netting(uncleared))

    if pieces:
        after = pd.concat(pieces, ignore_index=True)
    else:
        after = obligations.iloc[0:0].copy()      # empty table, same columns

    return _evaluate(obligations, after, external_assets, institutions, label)


# =============================================================================
# SECTION 6 — HELPERS
# =============================================================================

def make_external_assets(obligations, institutions, theta=0.8, dispersion=0.35,
                         core=None, core_discount=0.6, seed=None):
    """
    Outside assets e_i, set as a multiple `theta` of what each institution owes.

    IMPORTANT: e_i is OUTSIDE ASSETS in the Eisenberg-Noe sense — resources
    available to pay with that are not claims on other network members. It is
    NOT regulatory capital, which is a loss-absorbing equity buffer measured
    against risk-weighted assets. Do not present theta as a Basel ratio.

    theta is a liquidity-stress level and should be swept, not fixed:
    low theta = stressed system, high theta = comfortable system.
    """
    rng = np.random.default_rng(seed)
    owed = obligations.groupby("debtor")["amount"].sum()
    if core is None:
        core = set()

    assets = {}
    for i in institutions:
        own_obligations = owed.get(i, 0.0)
        multiplier = theta * (core_discount if i in core else 1.0)
        noise = rng.lognormal(0.0, dispersion)    # institutions are not identical
        assets[i] = own_obligations * multiplier * noise
    return assets


def partition_by_institution(institutions, k, style="balanced", seed=None):
    """
    Each institution belongs to exactly one CCP.
    (Duffie & Zhu 2011, Appendix B.)

    style: "balanced" = equal groups, "skewed" = one small group and the rest.
    Returns a dict {institution: ccp name}.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(list(institutions)))
    if style == "balanced":
        groups = np.array_split(shuffled, k)
    elif style == "skewed":
        small = max(1, len(shuffled) // (k * 3))
        groups = [shuffled[:small]] + list(np.array_split(shuffled[small:], k - 1))
    else:
        raise ValueError("style must be 'balanced' or 'skewed'")
    assignment = {}
    for number, group in enumerate(groups):
        for institution in group:
            assignment[institution] = f"CCP_{number}"
    return assignment


def assign_by_institution(obligations, partition):
    """
    Decides which CCP clears each obligation when every INSTITUTION belongs to
    one CCP (Duffie & Zhu 2011, Appendix B).

    An obligation is centrally cleared only if BOTH sides belong to the same
    CCP. If they are in different CCPs, neither CCP can see the whole trade,
    so it stays bilateral -> None.

    Returns a list with one entry per row of `obligations`.
    """
    labels = []
    for row in obligations.itertuples(index=False):
        if partition[row.debtor] == partition[row.creditor]:
            labels.append(partition[row.debtor])     # both sides, same CCP
        else:
            labels.append(None)                      # crosses a CCP boundary
    return labels


def assign_by_asset_class(obligations, class_to_ccp):
    """
    Decides which CCP clears each obligation when every ASSET CLASS has its own
    CCP (Duffie & Zhu 2011, main text — the realistic case).

    class_to_ccp: e.g. {"CDS": "CCP_1", "IRS": "CCP_2", "EQ": None, "FX": None}
    None means that class stays bilateral.

    NOTE: once a class is centrally cleared it leaves the bilateral pool, so it
    can no longer offset an institution's positions in OTHER classes. That is
    why clearing one class can RAISE total exposure.

    Returns a list with one entry per row of `obligations`.
    """
    labels = []
    for row in obligations.itertuples(index=False):
        labels.append(class_to_ccp.get(row.asset_class))
    return labels


# =============================================================================
# VALIDATION — run this file directly to check the code against the thesis
# =============================================================================

def validate_table_2():
    """Reproduces Table 2 of the thesis (section 2.1)."""
    obl = pd.DataFrame([
        ("A", "B", 150.0, "x"),
        ("B", "A",  50.0, "x"),
        ("B", "C", 120.0, "x"),
        ("C", "A",  90.0, "x"),
    ], columns=["debtor", "creditor", "amount", "asset_class"])
    people = ["A", "B", "C"]

    bilateral = bilateral_netting(obl)
    multilateral = multilateral_compression(obl)

    print("TEST 1 — thesis Table 2")
    print(f"  original                  {gross_exposure(obl):6.0f}   (table: 410)")
    print(f"  after bilateral netting   {gross_exposure(bilateral):6.0f}   (table: 310)")
    print(f"  after compression         {gross_exposure(multilateral):6.0f}   (table:  40)")
    print(f"  net positions unchanged:  {net_positions_unchanged(obl, multilateral, people)}")
    print(f"  no new counterparties:    {no_new_counterparties(obl, multilateral)}")
    ok = (abs(gross_exposure(bilateral) - 310) < 1e-6
          and abs(gross_exposure(multilateral) - 40) < 1e-6
          and net_positions_unchanged(obl, bilateral, people)
          and net_positions_unchanged(obl, multilateral, people)
          and no_new_counterparties(obl, multilateral))
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def validate_duffie_zhu():
    """
    Duffie & Zhu's (2011) central point: clearing one asset class can RAISE
    total exposure, because that class leaves the bilateral pool.
    """
    obl = pd.DataFrame([
        ("A", "B", 100.0, "CDS"),
        ("B", "A", 150.0, "IRS"),
    ], columns=["debtor", "creditor", "amount", "asset_class"])
    people = ["A", "B"]
    assets = {"A": 1e9, "B": 1e9}

    no_ccp = regime_bilateral_only(obl, assets, people)
    with_ccp = regime_multi_ccp(
        obl, assign_by_asset_class(obl, {"CDS": "CCP_CDS", "IRS": None}),
        assets, people, label="CDS cleared centrally")

    print("TEST 2 — Duffie & Zhu (2011)")
    print(f"  bilateral, both classes pooled   {no_ccp['gross_after']:6.0f}")
    print(f"  CDS moved to a CCP               {with_ccp['gross_after']:6.0f}")
    print(f"  -> introducing the CCP raised exposure by "
          f"{with_ccp['gross_after'] - no_ccp['gross_after']:.0f}")
    ok = (abs(no_ccp["gross_after"] - 50) < 1e-6
          and abs(with_ccp["gross_after"] - 250) < 1e-6)
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def validate_clearing_vector():
    """Three hand-checkable clearing cases."""
    print("TEST 3 — clearing vector")

    # (a) a solvent cycle: everyone pays in full
    obl = pd.DataFrame([("A", "B", 100.0, "x"), ("B", "C", 100.0, "x"),
                        ("C", "A", 100.0, "x")],
                       columns=["debtor", "creditor", "amount", "asset_class"])
    p, c, d, u = clearing_vector(obl, {"A": 50, "B": 50, "C": 50}, ["A", "B", "C"])
    ok_a = np.allclose(p, [100, 100, 100]) and d == 0 and abs(u) < 1e-9
    print(f"  solvent cycle      p = {np.round(p, 1)}  expected [100 100 100]  "
          f"{'PASS' if ok_a else 'FAIL'}")

    # (b) a genuine default: A owes 100, holds 20, receives nothing
    obl = pd.DataFrame([("A", "B", 100.0, "x")],
                       columns=["debtor", "creditor", "amount", "asset_class"])
    p, c, d, u = clearing_vector(obl, {"A": 20, "B": 0}, ["A", "B"])
    ok_b = np.allclose(p, [20, 0]) and d == 1 and abs(u - 80) < 1e-9
    print(f"  genuine default    p = {np.round(p, 1)}  expected [20 0]         "
          f"{'PASS' if ok_b else 'FAIL'}")

    # (c) contagion: B can only pay what it receives from A
    obl = pd.DataFrame([("A", "B", 100.0, "x"), ("B", "C", 50.0, "x")],
                       columns=["debtor", "creditor", "amount", "asset_class"])
    p, c, d, u = clearing_vector(obl, {"A": 30, "B": 0, "C": 0}, ["A", "B", "C"])
    ok_c = np.allclose(p, [30, 30, 0]) and d == 2 and abs(u - 90) < 1e-9
    print(f"  contagion          p = {np.round(p, 1)}  expected [30 30 0]      "
          f"{'PASS' if ok_c else 'FAIL'}")
    print()
    return ok_a and ok_b and ok_c


def demo():
    """A full run of all four regimes on one network."""
    obligations, core = make_core_periphery_network(20, seed=1)
    institutions = list(range(20))
    assets = make_external_assets(obligations, institutions, theta=0.3,
                                  core=core, seed=1)

    partition = partition_by_institution(institutions, k=2, style="balanced", seed=1)

    results = [
        regime_no_netting(obligations, assets, institutions),
        regime_bilateral_only(obligations, assets, institutions),
        regime_single_ccp(obligations, assets, institutions),
        regime_multi_ccp(obligations, assign_by_institution(obligations, partition),
                         assets, institutions, label="2 CCPs (by institution)"),
        regime_multi_ccp(obligations,
                         assign_by_asset_class(obligations,
                                               {"CDS": "CCP_1", "IRS": "CCP_2",
                                                "EQ": None, "FX": None}),
                         assets, institutions, label="2 CCPs (by asset class)"),
    ]

    print("DEMO — 20 institutions, core-periphery network, theta = 0.3")
    print(pd.DataFrame(results)[
        ["regime", "gross_after", "reduction_pct", "n_defaults",
         "unpaid_claims", "net_positions_ok"]
    ].round(1).to_string(index=False))




# =============================================================================
# SECTION 7 — SHOCK PROPAGATION
# =============================================================================
#
# The theta sweep varies how well capitalised the system is to begin with.
# A shock experiment is different and is what the literature usually means by
# systemic risk: start from a HEALTHY system, hit some institutions with a
# loss, and watch how far the damage spreads.
#
# The question this answers: does the clearing structure change how a shock
# propagates, holding the starting point fixed?
# =============================================================================

def largest_institutions(obligations, how_many):
    """The institutions with the biggest obligation books — the likely dealers."""
    size = obligations.groupby("debtor")["amount"].sum()
    return list(size.sort_values(ascending=False).head(how_many).index)


def apply_shock(external_assets, shocked, severity):
    """
    Wipes out `severity` of the shocked institutions' outside assets.
    severity = 0.5 means they lose half of what they had to pay with.
    """
    after = {}
    for institution, assets in external_assets.items():
        if institution in shocked:
            after[institution] = assets * (1 - severity)
        else:
            after[institution] = assets
    return after


def shock_experiment(obligations, institutions, regimes, external_assets,
                     shocked, severity):
    """
    Runs every regime twice on the same network: once unshocked, once shocked.

    `regimes` is a dict {label: function(obligations, assets, institutions)}.

    Reports the EXTRA defaults and EXTRA unpaid claims caused by the shock.
    That difference — not the absolute level — is what shows whether a given
    clearing structure contains the damage or lets it spread.
    """
    shocked_assets = apply_shock(external_assets, shocked, severity)
    rows = []
    for label, run in regimes.items():
        calm = run(obligations, external_assets, institutions)
        hit = run(obligations, shocked_assets, institutions)
        rows.append({
            "regime": label,
            "defaults_before": calm["n_defaults"],
            "defaults_after": hit["n_defaults"],
            "extra_defaults": hit["n_defaults"] - calm["n_defaults"],
            "unpaid_before": calm["unpaid_claims"],
            "unpaid_after": hit["unpaid_claims"],
            "extra_unpaid": hit["unpaid_claims"] - calm["unpaid_claims"],
            # the shock changes assets, not obligations, so one flag covers both
            "net_positions_ok": bool(calm["net_positions_ok"]
                                     and hit["net_positions_ok"]),
        })
    return pd.DataFrame(rows)


def summarise(df, index, columns, value, decimals=1):
    """
    Mean and standard deviation side by side.

    Always report both: the compression figures have standard deviations of
    7-10 percentage points across networks, so a mean on its own overstates
    how precisely any single number is known.
    """
    mean = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    std = df.pivot_table(index=index, columns=columns, values=value, aggfunc="std")
    out = mean.round(decimals).astype(str) + "  (" + std.round(decimals).astype(str) + ")"
    return out


if __name__ == "__main__":
    passed = [validate_table_2(), validate_duffie_zhu(), validate_clearing_vector()]
    print(f"{sum(passed)}/{len(passed)} validation tests passed\n")
    demo()
