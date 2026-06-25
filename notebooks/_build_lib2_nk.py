"""Build lib2_nk.gmt — the NK-cell counterpart of lib2_{cd8,cd4,bcell,monocyte}.gmt.

Same composition/format as the existing lib2 files:
  1. ~12 curated HC_NK_* marker programs (cytotoxic / cytokine / IFN / etc.).
  2. MSigDB C7 IMMUNESIGDB NK gene sets (term name contains NK / NATURAL_KILLER).

The C7 sets are harvested from the lib*.gmt files already in this folder (the exact
same MSigDB content + gene symbols used by the other libraries -> deterministic,
offline-safe). If gseapy can reach MSigDB it tops the harvest up with the full
c7.immunesigdb NK collection; failure is non-fatal (harvest alone suffices).

GMT format: term<TAB>description<TAB>gene1<TAB>gene2...

Run:  python notebooks/_build_lib2_nk.py
Writes: notebooks/lib2_nk.gmt
"""
from __future__ import annotations
import re
from pathlib import Path

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
OUT_PATH = NB_DIR / "lib2_nk.gmt"
SOURCE_GMTS = ["lib1_immune", "lib2_cd8", "lib2_cd4", "lib2_bcell", "lib2_monocyte"]
NK_RE = re.compile(r"NK|NATURAL_KILLER", re.IGNORECASE)
MIN_GENES = 5

# Curated NK marker programs (human symbols). Mirrors the HC_<CT>_<PROGRAM> sets in
# the other lib2 files; canonical NK biology.
HC_NK = {
    "HC_NK_CYTOTOXIC_CD56DIM": [
        "GNLY", "PRF1", "GZMB", "GZMH", "NKG7", "KLRD1", "FGFBP2", "FCGR3A",
        "SPON2", "CX3CR1", "S1PR5", "PLEK", "CST7", "CTSW", "KLRF1",
    ],
    "HC_NK_CYTOKINE_CD56BRIGHT": [
        "GZMK", "SELL", "XCL1", "XCL2", "IL7R", "GPR183", "KLRC1", "TCF7",
        "CD44", "CAPG", "COTL1", "IL2RB", "CCR7", "LTB", "TNFRSF18",
    ],
    "HC_NK_IFN_RESPONSE": [
        "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2",
        "OASL", "STAT1", "IRF7", "IFI6", "IFI44L", "RSAD2", "XAF1",
    ],
    "HC_NK_ACTIVATION": [
        "NCR1", "NCR3", "KLRK1", "CD69", "TNFRSF9", "CRTAM", "IFNG", "TNF",
        "CCL3", "CCL4", "MKI67", "LAMP1", "TNFRSF18", "ICOS", "HLA-DRA",
    ],
    "HC_NK_INHIBITORY_RECEPTORS": [
        "KIR2DL1", "KIR2DL3", "KIR3DL1", "KIR3DL2", "KLRC1", "LAG3", "TIGIT",
        "KLRB1", "HAVCR2", "CD96", "LILRB1", "KLRG1", "SIGLEC7", "CD160", "PDCD1",
    ],
    "HC_NK_ADAPTIVE_MEMORY": [
        "KLRC2", "IL32", "ZBTB16", "GZMH", "CD3E", "FCER1G", "SYK", "B3GAT1",
        "CD52", "PATL2", "LILRB1", "CADM1", "KLRC3", "IKZF2", "ZNF683",
    ],
    "HC_NK_PROLIFERATION": [
        "MKI67", "TOP2A", "STMN1", "TYMS", "TUBB", "TUBA1B", "PCNA", "CDK1",
        "CCNB1", "CCNB2", "UBE2C", "BIRC5", "CENPF", "RRM2", "HMGB2",
    ],
    "HC_NK_CHEMOKINE": [
        "CCL3", "CCL4", "CCL5", "XCL1", "XCL2", "CXCR4", "CXCR3", "CCL3L1",
        "CCL4L2", "CXCL8", "CCR1", "CCR5", "CX3CR1", "S1PR5", "SELPLG",
    ],
    "HC_NK_MATURATION": [
        "B3GAT1", "ZEB2", "TBX21", "PRDM1", "S1PR5", "CX3CR1", "FCGR3A",
        "KLRG1", "GZMB", "PRF1", "ZNF683", "CXCR1", "CXCR2", "FGFBP2", "ADGRG1",
    ],
    "HC_NK_TISSUE_RESIDENT": [
        "ITGA1", "ITGAE", "CXCR6", "ZNF683", "CD69", "ITGAD", "RGS1", "DUSP6",
        "NR4A1", "NR4A2", "NR4A3", "CD49A", "AREG", "TNFRSF18", "GZMK",
    ],
    "HC_NK_RECEPTOR_REPERTOIRE": [
        "KLRD1", "KLRC1", "KLRC2", "KLRC3", "KLRF1", "KLRB1", "KLRK1", "NCR1",
        "NCR2", "NCR3", "CD244", "SLAMF7", "CD226", "FCGR3A", "TYROBP",
    ],
    "HC_NK_METABOLISM_STRESS": [
        "FOS", "FOSB", "JUN", "JUNB", "DUSP1", "DUSP2", "HSPA1A", "HSPA1B",
        "ZFP36", "KLF6", "BTG1", "BTG2", "PPP1R15A", "NR4A1", "GADD45B",
    ],
}


def _parse_gmt(path: Path):
    out = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        out[parts[0]] = [g for g in parts[2:] if g]
    return out


def _harvest_local():
    nk = {}
    for stem in SOURCE_GMTS:
        p = NB_DIR / f"{stem}.gmt"
        if not p.exists():
            continue
        for term, genes in _parse_gmt(p).items():
            if NK_RE.search(term) and term not in nk:
                nk[term] = genes
    print(f"harvested {len(nk)} NK terms from local lib*.gmt", flush=True)
    return nk


def _gseapy_topup():
    try:
        from gseapy import Msigdb
        msig = Msigdb()
        gmt = None
        for dbver in ("2024.1.Hs", "2023.2.Hs", "2023.1.Hs"):
            try:
                gmt = msig.get_gmt(category="c7.immunesigdb", dbver=dbver)
                if gmt:
                    print(f"gseapy: fetched c7.immunesigdb {dbver}", flush=True)
                    break
            except Exception:
                continue
        if not gmt:
            return {}
        nk = {t: list(g) for t, g in gmt.items() if NK_RE.search(t)}
        print(f"gseapy: {len(nk)} NK terms from full c7.immunesigdb", flush=True)
        return nk
    except Exception as exc:
        print(f"gseapy top-up skipped ({exc})", flush=True)
        return {}


def main():
    terms = dict(HC_NK)  # curated first; take precedence on name collision
    harvest = _harvest_local()
    topup = _gseapy_topup()
    for src in (topup, harvest):  # harvest last so its exact symbols win for shared names
        for t, g in src.items():
            terms[t] = g

    lines = []
    for term, genes in terms.items():
        genes = list(dict.fromkeys(g for g in genes if g))  # dedup, keep order
        if len(genes) < MIN_GENES:
            continue
        desc = "curated" if term.startswith("HC_NK_") else "C7_IMMUNESIGDB"
        lines.append("\t".join([term, desc, *genes]))

    OUT_PATH.write_text("\n".join(lines) + "\n")
    n_curated = sum(1 for l in lines if l.startswith("HC_NK_"))
    print(f"wrote {OUT_PATH}: {len(lines)} terms ({n_curated} curated HC_NK_*, "
          f"{len(lines) - n_curated} C7)", flush=True)


if __name__ == "__main__":
    main()
