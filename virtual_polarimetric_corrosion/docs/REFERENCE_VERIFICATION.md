# Reference verification pass (2026-10-02)

**Method and limits.** Web search only. `doi.org`, `api.crossref.org` and publisher pages are blocked by the egress proxy, so no DOI was resolved against Crossref and no full text was read. "Confirmed" below means a search result agreed on authors, venue and year; it does not mean the paper's content was checked.

## Primary paper

Yu, Z.; Wang, D.; Wu, H. (2025). *Defect Detection Method for Large-Curvature and Highly Reflective Surfaces Based on Polarization Imaging and Improved YOLOv11.* Photonics 12(4):368. DOI 10.3390/photonics12040368.
- Confirmed: title, authors, journal, volume/issue/article number, 2025, affiliation (Zhejiang Normal University); MDPI URL https://www.mdpi.com/2304-6732/12/4/368 appears in search results.
- Not confirmed: the DOI string itself (derived from the MDPI URL pattern, not resolved) and the reported metrics (P 86.1 %, R 71.1 %, mAP50 72.7 %), which remain abstract-level snippets.

## Reference list

| Key | Reference | DOI / id | Status |
|---|---|---|---|
| yu2025photonics | Yu, Wang, Wu 2025, Photonics 12(4):368 | 10.3390/photonics12040368 | Confirmed (metadata); DOI unresolved |
| ba2020deepsfp | Ba et al., Deep Shape from Polarization, ECCV 2020 | 10.1007/978-3-030-58586-0_33 | Confirmed: authors, venue, DOI shown in search results (ECVA, UCLA, arXiv 1903.10210) |
| nayar1997sep | Nayar, Fang, Boult, IJCV 21(3):163-186, 1997 | 10.1023/A:1007937815113 | Confirmed: authors, volume, pages, DOI; issue number 3 not independently seen |
| sharma2005de2000 | Sharma, Wu, Dalal, Color Res. Appl. 30(1):21-30, 2005 | 10.1002/col.20070 | Confirmed (author site and DOI link in results). Upgrade from `from_memory` |
| rustseg2022 | Burton, Nash, Birbilis, RustSEG, arXiv 2205.05426 | arXiv:2205.05426 | Authors now found (fills the "authors not retrieved" gap); arXiv listing seen in search |
| jei2019corr | Segmenting localized corrosion from rust-removed metallic surface..., J. Electron. Imaging 28(4):043019 | 10.1117/1.JEI.28.4.043019 | Confirmed (SPIE URL in results); DOI derived from the URL, authors still not retrieved |
| ievpf2025 | Virtual Polarization Filtering ... Additive Manufacturing, Photonics 12(6):599 | not retrieved | Venue confirmed (MDPI URL /2304-6732/12/6/599); resolves the "to confirm" note. Authors not retrieved |
| heritage2023 | U-Net corrosion-compound segmentation on iron/copper heritage microscopy (Oltenia Museum) | not retrieved | Existence and content confirmed via riuma.uma.es repository results; authors, title, venue, DOI still missing |
| porosity2025 | Micro-scale porosity in reflective metal parts via polarization imaging | PMC12156235 | Title and PMC id confirmed |

Not rechecked in this pass (stay at their existing status in `BIBLIOGRAPHY.csv`): lei2022spw, zhao2020pmvir, atkinson2006diffuse, baek2018/2020, hwang2022, dave2022pandora, li2024neisf, neisfpp2024, kadambi2015p3d, kalra2020deeppol, kajiyama2023sep, lin2025rgb2pol, genpolar2026, pitting2024, mcdnet2024, johnson1972, cook1982, walter2007, shafer1985, wolff1991, born1999optics, chipman2018, goldstein2011, gal2016dropout, lakshminarayanan2017, kendall2017, hinton2015kd. Every `from_memory` entry here is still unverified.

## Claims lacking a source

- Any "novel" / "first" wording for the 8 architectures in `corrosion-research/RESEARCH.md` §3-4: the novelty audit is self-assessed; §6 search matrix was not executed (see below).
- The novelty-ranking scores and "overlap risk" column are the design team's judgement, not evidence.
- No source in the repo supports a polarization-feature benefit on real copper/bronze corrosion; `README.md` already states no real polarization data exists. Closest evidence is on other surfaces (Yu 2025, Kalra 2020, Photonics 2025 porosity).
- Copper/bronze-specific optical constants (Johnson & Christy 1972) are cited from memory only.

## Literature and patent search (§6 of RESEARCH.md): status

Only a first pass on the polarization-corrosion theme was possible here. §6's per-architecture query matrix (PMD-Net, CPEN, HyperCorNet, CorroNCA, Corro-SSM, EB-TopoSeg, INCF, MoMER) was **not run**, and Google Patents, WIPO Patentscope, Scholar, IEEE Xplore and Scopus are not reachable by this search tool.

Findings so far:
- A query on "polarization imaging copper bronze corrosion patent" found no patent combining polarization imaging with copper/bronze corrosion assessment. Patents seen are generic: US 12,320,741 B2 (Sony, polarization imaging system and method), US 12,320,705 (polarization imaging device and method), US 8,786,755 B2, and a USPTO "surface defect measuring by microscopic scattering polarization imaging" application. Related and worth a closer read: US 12,217,408 (semantic deep learning and rule optimization for surface corrosion detection). None were read in full, so this is not a freedom-to-operate conclusion.
- Bronze-corrosion studies using polarised-light metallography exist (e.g. Manti & Watkinson, Cardiff, ORCA eprint 151142); they are microscopy, not computational imaging.
- Closest copper-corrosion segmentation precedent remains the Oltenia Museum U-Net work (no polarization).

## Suggested next steps

1. Allow-list `api.crossref.org` (or `doi.org`) so DOIs can be resolved in bulk.
2. Run the §6 matrix from a machine with Scholar/Patentscope access, or supply exports.
3. After that, update `BIBLIOGRAPHY.csv` statuses (needs your go-ahead; nothing was edited there).
