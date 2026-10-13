"""Acceptance rule for LLM-judged edit pairs (shared by scripts/08_cf_variants.py and 09_revision_analyses.py).

A pair is accepted when the judge says the edit did what it was meant to and nothing else of substance:
  nuisance / nuisance_rev : target same, nuisance changed
  target                  : target changed, nuisance same
  placebo                 : neither judged changed ('unclear' allowed: e.g. a bio that never reveals gender)
  placebo_group           : target not judged changed (the judge is asked *which* group is named, and Christian -> Protestant
                            does change that, while our binary nuisance -- Muslim/LGBTQ mentioned -- does not)
and, for every kind, no *major* other meaning change and no fluency degradation (the edited text is fluent, or the
source already was not: the judges rate many untouched informal reviews/comments as non-fluent).
"""
import numpy as np
import pandas as pd


def accept(j: pd.DataFrame) -> pd.Series:
    y = lambda c: j[c].astype(str).str.startswith("yes")  # noqa: E731
    n = lambda c: j[c].astype(str).str.startswith("no")  # noqa: E731
    base = (y("edit_fluent") | n("src_fluent")) & ~j.other_change.astype(str).str.startswith("major")
    v = np.select(
        [j.kind.isin(["nuisance", "nuisance_rev"]), j.kind.eq("target"), j.kind.eq("placebo"), j.kind.eq("placebo_group")],
        [y("target_same") & n("nuis_same"), n("target_same") & y("nuis_same"), ~n("target_same") & ~n("nuis_same"),
         ~n("target_same")], False)
    return pd.Series(v & base.values, index=j.index)
