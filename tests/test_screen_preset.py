"""SCREEN_TASKS: the triage set for "is this checkpoint better or worse, and where".

Chosen by measured discriminative spread across five same-family checkpoints, divided by
sequences to embed, then one or two per domain. Two rules the set has to keep:

  * every task needs a test set big enough that a delta means something. The three
    highest raw spreads in the suite -- flip2_rhomax (n=184), contact_probe (n=40),
    ppi_affinity (n=200) -- rank top precisely BECAUSE small test sets turn noise into
    spread. A screen built on them raises false alarms, which is the one thing a screen
    must not do.
  * every task has to actually move. signalp_binary spread 0.0018 across models that
    differ by 0.1363 on ec_classification: it cannot discriminate, so it is pure cost.
"""

from __future__ import annotations

from benchmark_tasks import SCREEN_TASKS, TASKS

# n(test) for the assays too small to carry a conclusion.
TOO_SMALL = {"contact_probe": 40, "cath_eat": 150, "flip2_rhomax": 184,
             "deepet_topt": 185, "ppi_affinity": 200}
# Measured spread across v3 / v4-36k / Base-10k / ArmA-10k / ArmB-10k.
CANNOT_DISCRIMINATE = {"signalp_binary": 0.0018, "variant_effect": 0.0055,
                       "subcellular_loc": 0.0056, "profet_np_sp_cleaved": 0.0057}


def test_every_screen_task_exists():
    unknown = [t for t in SCREEN_TASKS if t not in TASKS]
    assert not unknown, f"SCREEN_TASKS names tasks the suite does not define: {unknown}"


def test_screen_excludes_small_test_sets():
    bad = sorted(set(SCREEN_TASKS) & set(TOO_SMALL))
    assert not bad, f"{bad} have test sets under 250; their spread is noise, not signal"


def test_screen_excludes_tasks_that_cannot_discriminate():
    bad = sorted(set(SCREEN_TASKS) & set(CANNOT_DISCRIMINATE))
    assert not bad, f"{bad} barely move between checkpoints -- cost without signal"


def test_screen_spans_the_domains():
    """A screen that only covers one domain answers 'better where?' with silence."""
    domains = {
        "structure": {"scope40_retrieval", "remote_homology"},
        "function": {"ec_classification"},
        "fitness": {"rhla_enzyme_mutations", "aav_flip"},
        "disorder": {"disprot"},
        "biophysics": {"stability", "conservation_flip"},
    }
    missing = [d for d, ts in domains.items() if not ts & set(SCREEN_TASKS)]
    assert not missing, f"screen covers no {missing} task"


def test_screen_stays_small():
    """It is a triage set. If it grows past ~10 it stops being faster than --fast."""
    assert len(SCREEN_TASKS) <= 10, f"{len(SCREEN_TASKS)} tasks is not a screen"

