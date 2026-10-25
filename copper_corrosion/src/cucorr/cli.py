"""Command-line interface: ``python -m cucorr.cli <command>`` or ``cucorr <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, deep_merge, load_config, resolve
from .utils.logging import get_logger

log = get_logger()


def cmd_audit(a):
    from .data.workbooks import audit
    from .reporting.audit_report import write_audit
    cfg = load_config(a.config)
    rep = audit(resolve(cfg, "settings_xlsx"), resolve(cfg, "labels_xlsx"))
    out = Path(a.out)
    write_audit(rep, out)
    log.info("audit written to %s (%d warnings)", out, len(rep["warnings"]))


def cmd_synth(a):
    from .data.synthetic import write_dataset
    write_dataset(a.out, a.n, a.seed)
    log.info("synthetic dataset written to %s", a.out)


def _load_dir(images_dir, masks_dir=None):
    from .utils.io import list_images, read_mask, read_rgb
    out = []
    for p in list_images(images_dir):
        m = None
        if masks_dir:
            mp = Path(masks_dir) / (p.stem + ".png")
            if not mp.exists():
                continue
            m = read_mask(mp)
        out.append((p.stem, read_rgb(p), m))
    return out


def _factories(names, cfg, exps=None):
    from .segmentation.registry import build
    fac = {}
    for n in names:
        if exps and n in exps:
            e = exps[n]
            c = deep_merge(cfg, {k: v for k, v in e.items() if k != "approach"})
            fac[n] = (lambda e=e, c=c: build(e["approach"], c))
        else:
            fac[n] = (lambda n=n: build(n, cfg))
    return fac


def cmd_predict(a):
    """Run one approach on a folder of images (approach A needs no training masks)."""
    from .reporting.outputs import save_annotation_outputs
    from .segmentation.registry import build
    from .utils.io import file_sha256, list_images, read_rgb
    cfg = load_config(a.config)
    seg = build(a.approach, cfg)
    if seg.requires_training:
        if not a.train_masks:
            sys.exit(f"{a.approach} needs --train-images/--train-masks")
        tr = _load_dir(a.train_images, a.train_masks)
        seg.fit([t[1] for t in tr], [t[2] for t in tr])
    for p in list_images(a.images or resolve(cfg, "images_dir")):
        img = read_rgb(p)
        mask, unc, proba = seg.predict_mask(img)
        save_annotation_outputs(Path(a.out) / a.approach / p.stem, img, mask, unc, proba, image_id=p.stem,
                                approach=seg.describe(), cfg=cfg, input_hash=file_sha256(p))
        log.info("%s: %.1f%% uncertain", p.name, 100 * (mask == 255).mean())


def cmd_benchmark(a):
    import pandas as pd
    import yaml
    from .data.splits import check_no_leakage, group_split
    from .evaluation.benchmark import run_benchmark
    cfg = load_config(a.config)
    exps = yaml.safe_load(open(a.experiments))["experiments"] if a.experiments else None
    data = _load_dir(a.images, a.masks)
    if a.splits and Path(a.splits).exists():
        sp = pd.read_csv(a.splits)
        if sp["split"].isna().any():
            sp = group_split(sp.image_id, sp.group_id, seed=cfg["seed"])
    else:
        sp = group_split([d[0] for d in data], [d[0] for d in data], seed=cfg["seed"])
        log.warning("no group file: each image treated as its own specimen (leakage risk if not true)")
    check_no_leakage(sp)
    names = a.approaches.split(",")
    rep = run_benchmark(_factories(names, cfg, exps), data, dict(zip(sp.image_id, sp.split)), cfg, a.out,
                        reference_type=a.reference_type, write_outputs=not a.no_outputs,
                        robustness=not a.no_robustness)
    print((Path(a.out) / "benchmark.md").read_text())
    _ = rep


def cmd_iaa(a):
    from .evaluation.agreement import calculate_inter_annotator_agreement
    from .utils.io import read_mask
    raters = {}
    for d in a.rater_dirs:
        d = Path(d)
        raters[d.name] = {p.stem: read_mask(p) for p in sorted(d.glob("*.png"))}
    common = sorted(set.intersection(*(set(v) for v in raters.values())))
    rep = calculate_inter_annotator_agreement({r: [v[i] for i in common] for r, v in raters.items()},
                                              iters=a.iters)
    rep["image_ids"] = common
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for k, v in rep["pairs"].items():
        print(f"{k}: kappa={v['cohen_kappa']:.3f} CI={v['cohen_kappa_ci']} binary_kappa={v['binary_kappa']:.3f}")


def cmd_active(a):
    from .active_learning.loop import run_active_learning
    from .data.splits import group_split
    from .segmentation.registry import build
    cfg = load_config(a.config)
    f = cfg["approach_f"]
    data = _load_dir(a.images, a.masks)
    sp = group_split([d[0] for d in data], [d[0] for d in data], (0.7, 0.3, 0.0), cfg["seed"])
    split = dict(zip(sp.image_id, sp.split))
    pool = [d for d in data if split[d[0]] == "train"]
    val = [d for d in data if split[d[0]] == "val"]
    results = {}
    for strat in ("random", "uncertainty", "disagreement"):
        rng = np.random.default_rng(cfg["seed"])
        init = rng.choice(len(pool), min(f["budget_per_round"], len(pool)), replace=False).tolist()
        results[strat] = run_active_learning(
            lambda: build("approach_b", cfg), [p[1] for p in pool], lambda i: pool[i][2],
            [v[1] for v in val], [v[2] for v in val], strategy=strat, budget_per_round=f["budget_per_round"],
            max_rounds=f["max_rounds"], init_idx=init, min_improvement=f["min_improvement"],
            patience=f["patience"], seed=cfg["seed"],
            make_committee=lambda: [build(c, cfg) for c in f["components"]])
        log.info("%s: %s", strat, results[strat]["curve"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cucorr")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit"); s.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "audit"))
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("synth"); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=12)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_synth)
    s = sub.add_parser("predict"); s.add_argument("--approach", default="approach_a")
    s.add_argument("--images"); s.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    s.add_argument("--train-images"); s.add_argument("--train-masks"); s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("benchmark"); s.add_argument("--images", required=True); s.add_argument("--masks", required=True)
    s.add_argument("--splits", help="CSV image_id,group_id,split"); s.add_argument("--out", required=True)
    s.add_argument("--approaches", default="approach_a,approach_b,approach_c,approach_f")
    s.add_argument("--experiments", help="experiments.yaml; names in --approaches may be experiment keys")
    s.add_argument("--reference-type", required=True,
                   choices=["adjudicated_expert", "single_expert", "pseudo", "synthetic"])
    s.add_argument("--no-outputs", action="store_true"); s.add_argument("--no-robustness", action="store_true")
    s.set_defaults(fn=cmd_benchmark)
    s = sub.add_parser("iaa"); s.add_argument("rater_dirs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--iters", type=int, default=1000); s.set_defaults(fn=cmd_iaa)
    s = sub.add_parser("active-learning"); s.add_argument("--images", required=True)
    s.add_argument("--masks", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_active)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
