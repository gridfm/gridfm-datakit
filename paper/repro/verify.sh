#!/usr/bin/env bash
#
# Regenerate paper figures 2, 3, 4, 5 and 8 and compare every panel against the
# published PDFs in paper/extracted/figures/comparison_plots/.
#
# PDFs embed a creation timestamp, so their bytes never match. We rasterise both
# sides with pdftoppm and compare the pixels instead.
#
# Usage:  bash paper/repro/verify.sh [output_dir]
# Exits non-zero if any panel differs.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/out}"
REF="$REPO/paper/extracted/figures/comparison_plots"
DPI=100

if ! command -v pdftoppm >/dev/null 2>&1; then
    echo "error: pdftoppm not found (brew install poppler)" >&2
    exit 2
fi
if [ ! -d "$REF" ]; then
    echo "error: reference figures not found at $REF" >&2
    exit 2
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

echo "== Generating figures into $OUT"
python "$HERE/plot_spider.py"         --mode pf  --metric entropy --output-dir "$OUT" >/dev/null || exit 1
python "$HERE/plot_spider.py"         --mode opf --metric entropy --output-dir "$OUT" >/dev/null || exit 1
python "$HERE/plot_bar_branch.py"     --metric entropy            --output-dir "$OUT" >/dev/null || exit 1
python "$HERE/plot_violin.py"         --mode pf                   --output-dir "$OUT" >/dev/null || exit 1
python "$HERE/plot_violin.py"         --mode opf                  --output-dir "$OUT" >/dev/null || exit 1
python "$HERE/plot_branch_loading.py"                             --output-dir "$OUT" || exit 1

PANELS=(
    spider_plot_entropy_pf spider_plot_entropy_opf          # Figure 2
    Qg_violin_pf                                            # Figure 3
    barplot_branch_entropy_pf                               # Figure 4
    branch_loading_datakit branch_loading_pfdelta           # Figure 5
    Pd_violin_pf Qd_violin_pf Pg_violin_pf Vm_violin_pf Va_violin_pf          # Figure 8 (PF)
    Pd_violin_opf Qd_violin_opf Pg_violin_opf Qg_violin_opf Vm_violin_opf Va_violin_opf  # Figure 8 (OPF)
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo
echo "== Comparing $((${#PANELS[@]})) panels against $REF"
fail=0
for name in "${PANELS[@]}"; do
    mine="$OUT/$name.pdf"
    ref="$REF/$name.pdf"

    if [ ! -f "$mine" ]; then
        printf '%-12s %s (not generated)\n' "MISSING" "$name"; fail=1; continue
    fi
    if [ ! -f "$ref" ]; then
        printf '%-12s %s (no reference)\n' "SKIP" "$name"; continue
    fi

    pdftoppm -r "$DPI" -png "$ref"  "$TMP/ref_$name"  2>/dev/null
    pdftoppm -r "$DPI" -png "$mine" "$TMP/mine_$name" 2>/dev/null

    a="$(md5 -q "$TMP/ref_$name-1.png"  2>/dev/null || md5sum "$TMP/ref_$name-1.png"  2>/dev/null | cut -d' ' -f1)"
    b="$(md5 -q "$TMP/mine_$name-1.png" 2>/dev/null || md5sum "$TMP/mine_$name-1.png" 2>/dev/null | cut -d' ' -f1)"

    if [ -n "$a" ] && [ "$a" = "$b" ]; then
        printf '%-12s %s\n' "IDENTICAL" "$name"
    else
        printf '%-12s %s\n' "DIFFERS" "$name"; fail=1
    fi
done

echo
if [ "$fail" -eq 0 ]; then
    echo "All ${#PANELS[@]} panels are pixel-identical to the published figures."
else
    echo "Some panels differ -- see above." >&2
fi
exit "$fail"
