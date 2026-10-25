# References

**Verification status.** Entries were compiled from the authors' knowledge of the
literature. In this session the DOIs were **not** resolved online. Run
`python scripts/verify_references.py` to resolve every DOI against doi.org/Crossref before
citing anything. Entries marked ⚠ need extra checking of page or volume details. Do not
add a reference without checking it first.

## Copper corrosion products and colour
| # | Reference | Relevance |
|---|---|---|
| C1 | Scott, D. A. (2002). *Copper and Bronze in Art: Corrosion, Colorants, Conservation*. Getty Publications. ISBN 978-0-89236-638-5 | Standard reference on copper corrosion minerals (cuprite, tenorite, nantokite, atacamite/paratacamite, brochantite) and how they look |
| C2 | Leygraf, C., Odnevall Wallinder, I., Tidblad, J., & Graedel, T. (2016). *Atmospheric Corrosion* (2nd ed.). Wiley. ISBN 978-1-118-76227-0 | How patina forms; the Cu2O → basic copper salts sequence |
| C3 | Graedel, T. E., Nassau, K., & Franey, J. P. (1987). Copper patinas formed in the atmosphere—I. Introduction. *Corrosion Science*, 27(7), 639–657. doi:10.1016/0010-938X(87)90047-2 ⚠ | Patina composition and colour development |

## Colour science
| # | Reference | Relevance |
|---|---|---|
| S1 | Luo, M. R., Cui, G., & Rigg, B. (2001). The development of the CIE 2000 colour-difference formula: CIEDE2000. *Color Research & Application*, 26(5), 340–350. doi:10.1002/col.1049 | ΔE2000 used in Approach A |
| S2 | Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. *Color Research & Application*, 30(1), 21–30. doi:10.1002/col.20070 | Test pairs used in `tests/test_color.py` |
| S3 | CIE 015:2018 *Colorimetry*, 4th ed. Commission Internationale de l'Éclairage. doi:10.25039/TR.015.2018 | Definition of CIELAB, D65 and the observer |
| S4 | Buchsbaum, G. (1980). A spatial processor model for object colour perception. *Journal of the Franklin Institute*, 310(1), 1–26. doi:10.1016/0016-0032(80)90058-7 | Gray-world normalisation |

## Classical and graph-based segmentation
| # | Reference | Relevance |
|---|---|---|
| G1 | Achanta, R., et al. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. *IEEE TPAMI*, 34(11), 2274–2282. doi:10.1109/TPAMI.2012.120 | Approach C |
| G2 | Felzenszwalb, P. F., & Huttenlocher, D. P. (2004). Efficient graph-based image segmentation. *IJCV*, 59(2), 167–181. doi:10.1023/B:VISI.0000022288.19776.77 | Classical region proposer (Approach E fallback) |
| G3 | Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). Learning with local and global consistency. *NeurIPS 16* | Graph label propagation / smoothing |
| G4 | Krähenbühl, P., & Koltun, V. (2011). Efficient inference in fully connected CRFs with Gaussian edge potentials. *NeurIPS 24* | Optional dense CRF refinement |

## Machine learning
| # | Reference | Relevance |
|---|---|---|
| M1 | Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. doi:10.1023/A:1010933404324 | Approach B/C |
| M2 | Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297. doi:10.1007/BF00994018 | Approach B |
| M3 | Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. doi:10.1214/aos/1013203451 | Approach B |
| M4 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*. doi:10.1145/2939672.2939785 | Gradient-boosting alternative |
| M5 | Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. *MICCAI 2015*, LNCS 9351. doi:10.1007/978-3-319-24574-4_28 | Approach D |
| M6 | Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV 2018*. doi:10.1007/978-3-030-01234-2_49 ⚠ | DeepLabV3+ |
| M7 | He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*. doi:10.1109/CVPR.2016.90 | ResNet encoders |
| M8 | Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net. *3DV*. doi:10.1109/3DV.2016.79 | Dice loss |
| M9 | Kirillov, A., et al. (2023). Segment Anything. *ICCV*. arXiv:2304.02643 | Promptable segmentation |
| M10 | Ravi, N., et al. (2024). SAM 2: Segment anything in images and videos. arXiv:2408.00714 | Approach E |

## Weak supervision, active learning, uncertainty
| # | Reference | Relevance |
|---|---|---|
| W1 | Zhou, Z.-H. (2018). A brief introduction to weakly supervised learning. *National Science Review*, 5(1), 44–53. doi:10.1093/nsr/nwx106 | Palette-level and pseudo-label supervision |
| W2 | Lee, D.-H. (2013). Pseudo-label: the simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshop on Challenges in Representation Learning* | Pseudo-labelling |
| W3 | Settles, B. (2009). *Active Learning Literature Survey*. CS Tech. Rep. 1648, University of Wisconsin–Madison | Active-learning strategies |
| W4 | Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee. *COLT '92*. doi:10.1145/130385.130417 | Disagreement-based review |
| W5 | Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*, PMLR 48, 1050–1059 | MC dropout (Approach D) |
| W6 | Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 30* | Ensemble uncertainty |
| W7 | Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70, 1321–1330 | ECE |
| W8 | Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 30* | Coverage–risk curves |

## Agreement, metrics, statistics, explainability
| # | Reference | Relevance |
|---|---|---|
| A1 | Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.*, 20(1), 37–46. doi:10.1177/001316446002000104 | Cohen's κ |
| A2 | Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378–382. doi:10.1037/h0031619 | Fleiss' κ |
| A3 | Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. *J. Clin. Epidemiol.*, 43(6), 543–549. doi:10.1016/0895-4356(90)90158-L | How κ behaves when classes are imbalanced |
| A4 | Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174. doi:10.2307/2529310 | Interpretation bands (use with caution) |
| A5 | Dice, L. R. (1945). Measures of the amount of ecologic association between species. *Ecology*, 26(3), 297–302. doi:10.2307/1932409 | Dice coefficient |
| A6 | Csurka, G., Larlus, D., & Perronnin, F. (2013). What is a good evaluation measure for semantic segmentation? *BMVC* | Boundary F1, per-image metrics |
| A7 | Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. *BMC Medical Imaging*, 15, 29. doi:10.1186/s12880-015-0068-x | HD95 and metric selection |
| A8 | Maier-Hein, L., Reinke, A., et al. (2024). Metrics reloaded: recommendations for image analysis validation. *Nature Methods*, 21, 195–212. doi:10.1038/s41592-023-02151-z | Choosing metrics |
| A9 | Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC | Cluster (image-level) bootstrap |
| A10 | Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. doi:10.2307/3001968 | Paired test |
| A11 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statist.*, 6(2), 65–70 | Multiple-comparison correction |
| A12 | Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30 | Comparing methods |
| A13 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30* | SHAP |
