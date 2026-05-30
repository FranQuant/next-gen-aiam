#!/usr/bin/env python3
"""
scripts/build_paper1.py — assemble paper1/sec*.md + paper1/app*.md into a
single pandoc-ready markdown and render to PDF via xelatex + citeproc.

Usage:
    python scripts/build_paper1.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "paper1" / "build"
OUT_MD = BUILD_DIR / "paper1.md"
OUT_PDF = BUILD_DIR / "paper1.pdf"
DOCS_PDF = ROOT / "docs" / "paper1.pdf"
BIB = ROOT / "docs" / "refs.bib"

MAIN_SECTIONS = [
    ROOT / "paper1/sec1_introduction.md",
    ROOT / "paper1/sec2_data_methods.md",
    ROOT / "paper1/sec3_vmp.md",
    ROOT / "paper1/sec4_estimator.md",
    ROOT / "paper1/sec5_regimes.md",
    ROOT / "paper1/sec6_costs_survival.md",
    ROOT / "paper1/sec7_conclusion.md",
]

APPENDIX_SECTIONS = [
    ROOT / "paper1/appA_universe.md",
    ROOT / "paper1/appB_strategy_zoo.md",
    ROOT / "paper1/appC_master_table.md",
    ROOT / "paper1/appD_regime_pipeline.md",
    ROOT / "paper1/appE_statistical_robustness.md",
    ROOT / "paper1/appF_transaction_costs.md",
    ROOT / "paper1/appG_subperiod.md",
    ROOT / "paper1/appH_reproducibility.md",
    ROOT / "paper1/glossary.md",
]

# Abstract text
ABSTRACT = (
    "We evaluate the principal families of classical portfolio allocation — mean-variance and "
    "maximum-Sharpe optimization, diversification objectives, risk parity, hierarchical risk "
    "parity, Black-Litterman, Fama-French factor tilts, time-series momentum, and "
    "regime-conditional switching — each tested across estimator choices (sample, Ledoit-Wolf, "
    "OAS) and with and without a volatility-managed overlay, yielding 62 strategy configurations "
    "on a single 29-asset multi-asset universe over 2003–2026 (23.3 years, monthly rebalance, "
    "252-day covariance lookback). Three patterns dominate, and they compose. Volatility "
    "management improves the Sharpe ratio on every base strategy "
    "(sign test $p \\approx 6 \\times 10^{-8}$). "
    "Robust covariance estimation matters significantly where the optimizer amplifies sample noise "
    "(the Maximum Sharpe Ratio family) but is near-irrelevant for Hierarchical Risk Parity, whose "
    "clustering already smooths block correlations. Regime conditioning, derived entirely from "
    "training data, adds a statistically significant lift (Memmel z = 2.05, p = 0.040). The "
    "strongest configuration combines all three: a volatility-managed regime-conditional switching "
    "rule, full-sample Sharpe 1.608. The cross-strategy ranking, however, is partly an artifact "
    "of sample-period luck — within-strategy variation across calendar years exceeds cross-strategy "
    "variation in the full-sample headline table, the paper's most defensible claim."
)

YAML_HEADER = f"""\
---
title: "Applied Portfolio Construction: A Multi-Asset Horse Race"
subtitle: "Volatility, Shrinkage, and Regimes Across Classical Strategies, 2003--2026"
author: "J. Francisco Salazar"
date: "Working Paper, May 2026"
abstract: |
  {ABSTRACT}
---

"""

# LaTeX header injected via --include-in-header
# - tcolorbox for transition paragraph tinting (RGB 230,240,250)
# - pdflscape for App G landscape tables
# - etoolbox \pretocmd for automatic \clearpage before every \section
#   (handles §1–§7, References, Appendices A–H, and Glossary in one rule)
# - \pretocmd{\tableofcontents}{\clearpage}{}{} puts TOC on its own page (p.2)
# - longtable/booktabs for wide tables; \setlength{\tabcolsep}{4pt} tightens columns
LATEX_HEADER = r"""\usepackage{longtable}
\usepackage{booktabs}
\usepackage{pdflscape}
\usepackage{etoolbox}
\usepackage{xcolor}
\raggedbottom
\setlength{\tabcolsep}{4pt}
\definecolor{transitbg}{RGB}{230,240,250}
\newsavebox{\transitioncontent}
\newenvironment{transitionbox}{%
  \begin{lrbox}{\transitioncontent}%
  \begin{minipage}{\dimexpr\linewidth-16pt\relax}%
  \setlength{\parindent}{0pt}%
  \setlength{\parskip}{6pt}%
}{%
  \end{minipage}%
  \end{lrbox}%
  \par\medskip\noindent
  \setlength{\fboxsep}{8pt}%
  \colorbox{transitbg}{\usebox{\transitioncontent}}%
  \par\medskip%
}
\pretocmd{\tableofcontents}{\clearpage}{}{}
\pretocmd{\section}{\clearpage}{}{}
"""

# Raw pandoc block for \clearpage (used explicitly between main body and appendices)
RAW_CLEARPAGE = "\n\n```{=latex}\n\\clearpage\n```\n\n"


def fix_citations(text: str) -> str:
    """Convert LaTeX \\citet{}/\\citep{} to pandoc [@key] format."""
    def citet_repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        return " and ".join(f"@{k}" for k in keys)

    def citep_repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        return "[" + "; ".join(f"@{k}" for k in keys) + "]"

    text = re.sub(r"\\citet\{([^}]+)\}", citet_repl, text)
    text = re.sub(r"\\citep\{([^}]+)\}", citep_repl, text)
    return text


def fix_table_names(text: str) -> str:
    """Abbreviate 'ledoit_wolf' → 'LW' inside markdown table rows only.
    Avoids modifying Python source code references in backtick spans."""
    lines = text.split("\n")
    result = []
    for line in lines:
        if line.strip().startswith("|"):
            line = line.replace("ledoit_wolf", "LW").replace(r"ledoit\_wolf", "LW")
        result.append(line)
    return "\n".join(result)


def fix_transition_divs(text: str) -> str:
    """Convert ::: transition ... ::: fenced divs into tcolorbox raw LaTeX blocks.
    The build script processes these before pandoc sees the markdown, so pandoc
    never encounters the fenced_divs syntax."""
    open_tag = "\n\n```{=latex}\n\\begin{transitionbox}\n```\n\n"
    close_tag = "\n\n```{=latex}\n\\end{transitionbox}\n```\n\n"

    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        return open_tag + inner + close_tag

    # Match ::: transition\n...\n::: (non-greedy, dotall)
    return re.sub(
        r"^::: transition\s*\n(.*?)\n:::[ \t]*$",
        repl,
        text,
        flags=re.MULTILINE | re.DOTALL,
    )


def load(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = fix_citations(text)
    text = fix_table_names(text)
    text = fix_transition_divs(text)
    return text


def assemble() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    parts = [YAML_HEADER]

    for f in MAIN_SECTIONS:
        parts.append(load(f).strip())
        parts.append("\n\n")

    # \clearpage before References (the etoolbox \pretocmd{\section} handles this
    # automatically because pandoc emits \section for level-1 headings, but we also
    # add an explicit one here for the unnumbered references heading which may not
    # trigger \pretocmd in all pandoc versions)
    parts.append(RAW_CLEARPAGE)
    parts.append("# References {.unnumbered}\n\n::: {#refs}\n:::\n\n")

    for f in APPENDIX_SECTIONS:
        parts.append(load(f).strip())
        parts.append("\n\n")

    assembled = "\n".join(parts)
    OUT_MD.write_text(assembled, encoding="utf-8")
    print(f"Assembled: {OUT_MD} ({len(assembled):,} chars, "
          f"{assembled.count(chr(10)):,} lines)")


def build_pdf() -> None:
    header_file = BUILD_DIR / "_header.tex"
    header_file.write_text(LATEX_HEADER, encoding="utf-8")

    cmd = [
        "pandoc", str(OUT_MD),
        "-o", str(OUT_PDF),
        "--pdf-engine=xelatex",
        "--citeproc",
        f"--bibliography={BIB}",
        "--variable", "geometry:margin=1in",
        "--variable", "fontsize=11pt",
        "--variable", "mainfont=Times New Roman",
        "--variable", "colorlinks=true",
        "--variable", "linkcolor=NavyBlue",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        f"--resource-path={ROOT / 'docs'}",
        f"--include-in-header={header_file}",
    ]
    print("Running pandoc...")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print("BUILD FAILED — stopping before docs copy.", file=sys.stderr)
        sys.exit(1)
    print(f"PDF built: {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


def copy_to_docs() -> None:
    shutil.copy(OUT_PDF, DOCS_PDF)
    print(f"Copied to: {DOCS_PDF}")


if __name__ == "__main__":
    assemble()
    build_pdf()
    copy_to_docs()
