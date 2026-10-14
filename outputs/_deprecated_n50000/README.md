# Deprecated outputs — do not cite

These two files were produced under the **withdrawn n = 50,000 coarse-granularity
convention** and are **not cited anywhere in the manuscript or the SI**. They are
kept only so that the replacement chain is auditable.

| File | Contains | Why it is wrong | Authoritative file |
|---|---|---|---|
| `confound_diagnostic.csv` | single-topology fraction 75.33% (coarse) / 97.97% (fine); fine entropy ratio 2.33% | n = 50,000 sample; superseded by the full-database run | `outputs/confound_full.csv` (**74.72%** coarse / **93.17%** fine; entropy ratio **17.99%** / **7.84%**) |
| `confound_diagnostic_pairs.csv` | pcu–nbo shared compositions **387**, Jaccard **0.358**; rht–tbo **1** | same n = 50,000 sample | `outputs/jaccard_full.csv` (**408**, **0.371**; rht–tbo **2**) |

**What to use instead**

- full-database confound statistics → `confound_full.csv`, produced by
  `scripts/full_db_confound.py`
- full-database topology-pair overlap → `jaccard_full.csv`, same script
- the per-held-out-topology chemistry seen/unseen decomposition → `chemistry_seen.csv`
  and `chemistry_seen_by_topology.csv`, produced by `scripts/chemistry_seen.py`

**Why these were removed from `outputs/` root.** A reader who opens the
reproduction package and greps for "pcu–nbo" previously found *two* mutually
inconsistent numbers (387 and 408). The n = 50,000 convention was retired before
first submission; leaving the files in the root directory put the retired numbers
back in front of a referee. See `REPRODUCE.md` for the convention history.

*Archived 2026-10-11 during revision round v6 (audit item P0/§6: "outputs 中仍存在
与正文冲突的陈旧 CSV").*
