Act as a senior computer-vision researcher, color-science specialist,
MATLAB deep-learning engineer, and scientific-software architect.

I am developing a reproducible pixel-level classification and corrosion
segmentation framework for copper-surface images. The classifier must
distinguish five mutually exclusive classes:

H  = Healthy copper
C1 = Copper(I) oxide, Cu2O
C2 = Copper(II) oxide, CuO
C3 = Copper(I) chloride, CuCl
C4 = Copper(II) chloride, CuCl2

The input feature vector is:

x = [L*, a*, b*, H, S, V, delta] ∈ R^7

where L*, a*, and b* are CIELAB coordinates; H, S, and V are HSV
coordinates; and delta is a color-distance or corrosion-deviation
feature.

First critically analyze the specification below. Identify and correct
any statistically invalid, scientifically questionable, ambiguous, or
technically unsupported requirements. Explain every correction briefly
before generating the implementation.

After the critical analysis, generate a complete, executable,
research-grade MATLAB project. Do not provide pseudocode, abbreviated
functions, omitted helper methods, TODO comments, or placeholders.

=======================================================================
1. SOFTWARE AND COMPATIBILITY
=======================================================================

Target MATLAB R2024b or newer, preferably the most recent supported
release.

Use modern MATLAB APIs wherever possible:

- trainnet
- dlnetwork
- featureInputLayer
- trainingOptions
- minibatchpredict
- exportONNXNetwork

Required products and add-ons should be documented explicitly:

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox Converter for ONNX Model Format

Optional components:

- Parallel Computing Toolbox for GPU acceleration
- Deep Learning Toolbox Model Compression Library for native MATLAB
  INT8 experiments
- Python with onnx, onnxruntime, and onnxruntime-tools or the current
  official ONNX Runtime quantization package

The project must detect missing optional dependencies and return
actionable messages rather than failing with obscure errors.

=======================================================================
2. PROJECT DELIVERABLES
=======================================================================

Generate a complete project with a clear directory structure, such as:

project/
│
├── main_train.m
├── main_inference.m
├── main_benchmark.m
├── config/
│   └── defaultConfig.m
├── data/
├── models/
├── outputs/
├── reports/
├── src/
│   ├── loadFeatureDataset.m
│   ├── validateFeatureDataset.m
│   ├── analyzeDeltaLeakage.m
│   ├── createStratifiedSplits.m
│   ├── createGroupStratifiedSplits.m
│   ├── fitMinMaxScaler.m
│   ├── applyMinMaxScaler.m
│   ├── buildCorrosionMLP.m
│   ├── computeClassWeights.m
│   ├── weightedCrossEntropyLoss.m
│   ├── trainCorrosionClassifier.m
│   ├── evaluateClassifier.m
│   ├── computeClassificationMetrics.m
│   ├── bootstrapMetricConfidenceIntervals.m
│   ├── exportModelArtifacts.m
│   ├── verifyONNXModel.m
│   ├── quantizeONNXModel.m
│   ├── extractImageFeatures.m
│   ├── computeDeltaFeature.m
│   ├── segmentCorrosionImage.m
│   ├── createSegmentationOverlay.m
│   ├── saveInferenceStatistics.m
│   └── benchmarkInference.m
├── python/
│   ├── quantize_onnx.py
│   ├── validate_onnx.py
│   └── requirements.txt
└── tests/
    ├── testDataValidation.m
    ├── testStratifiedSplit.m
    ├── testScaler.m
    ├── testMetrics.m
    ├── testFeatureExtraction.m
    └── testInferencePipeline.m

You may improve this structure, but every generated file must contain
complete working code.

=======================================================================
3. DATA INGESTION
=======================================================================

Accept either:

- CSV files using readtable
- Apache Parquet files using parquetread

Required columns:

L, a, b, H, S, V, delta, label

Required label values:

H, C1, C2, C3, C4

Optional metadata columns:

- image_id
- specimen_id
- sample_id
- acquisition_id
- x
- y
- source_file

Implement configurable column-name aliases, but internally convert all
columns to canonical names.

Examples:

- L, Lstar, L_star → L
- a, astar, a_star → a
- b, bstar, b_star → b
- class, category, target → label

The loader must:

1. Verify that every required column exists.
2. Convert all features to single precision after validation.
3. Convert labels into a categorical array with the fixed order:

   ["H", "C1", "C2", "C3", "C4"]

4. Reject unknown or missing labels.
5. Detect NaN, Inf, missing, or nonnumeric feature values.
6. Detect duplicated observations.
7. Detect identical feature vectors assigned conflicting labels.
8. Report the number and percentage of invalid rows.
9. Support configurable behavior:

   InvalidRowPolicy = "error" | "remove"

10. Save a data-quality report as both JSON and CSV.

=======================================================================
4. PHYSICAL RANGE VALIDATION
=======================================================================

Perform validation before normalization.

Default expected ranges:

- L*: [0, 100]
- a*: configurable, default warning range [-128, 127]
- b*: configurable, default warning range [-128, 127]
- S:  [0, 1]
- V:  [0, 1]
- delta: delta >= 0

Hue must support either:

- normalized hue in [0,1], or
- angular hue in [0,360)

Do not silently guess mixed hue conventions.

Provide:

HueConvention = "auto" | "normalized" | "degrees"

When HueConvention="auto":

- infer the likely convention from the data;
- reject datasets containing an apparent mixture of normalized and
  degree-based values;
- convert degree-based hue into [0,1];
- record the detected convention in the preprocessing metadata.

Range checking must distinguish:

- physically invalid values, which cause an error; and
- unusual but potentially valid values, which produce warnings.

=======================================================================
5. SCIENTIFIC HANDLING OF THE DELTA FEATURE
=======================================================================

Healthy pixels may currently contain delta=0. Do not assume this is
scientifically valid merely because it appears in the dataset.

The implementation must analyze whether delta creates target leakage.

Generate diagnostics including:

- class-wise minimum, maximum, mean, median, and standard deviation;
- percentage of zero delta values in each class;
- class histograms or density plots;
- mutual information or an equivalent univariate discriminative score;
- classification performance using delta alone;
- detection of the condition:

  all or nearly all H samples have delta=0 while nearly all corrosion
  samples have delta>0.

If this suspicious condition occurs, print a prominent warning that
delta may encode the target label rather than an independently
measurable property.

Support these modes:

DeltaMode = "provided" | "ciede2000" | "exclude"

A. DeltaMode="provided"

Use the dataset's delta column, but still perform leakage diagnostics.

B. DeltaMode="ciede2000"

Calculate delta as the minimum CIEDE2000 color difference between each
pixel and a configurable set of healthy CIELAB reference prototypes:

delta(x) = min_j ΔE00([L*,a*,b*], μ_H,j)

Implement the complete CIEDE2000 formula in MATLAB or use a verified
MATLAB function whose behavior is documented and tested.

The healthy references must come only from:

- externally supplied reference values, or
- the training partition.

Never calculate healthy prototypes using validation or test data.

C. DeltaMode="exclude"

Train a six-dimensional ablation model using:

[L*, a*, b*, H, S, V]

The primary experiment should train the requested seven-dimensional
model. However, if the leakage diagnostic is triggered, also train and
evaluate the six-dimensional ablation model and compare both models.

Report whether delta improves genuine generalization rather than merely
exploiting label encoding.

=======================================================================
6. REPRODUCIBLE STRATIFIED DATA SPLITTING
=======================================================================

Use the default fixed random seed:

RandomSeed = 42

Create train, validation, and test partitions using:

- 70% training
- 15% validation
- 15% testing

The proportions must be applied independently to every class. For
example, approximately 70% of H, 70% of C1, 70% of C2, 70% of C3, and
70% of C4 must enter the training set.

Use a deterministic largest-remainder or equivalent allocation method
to handle noninteger class counts.

Guarantee:

- no row appears in more than one partition;
- every class is represented in every partition when class size permits;
- class order is always H, C1, C2, C3, C4;
- split indices are reproducible;
- the split manifest is exported.

Preferred splitting strategy:

If image_id, specimen_id, sample_id, or acquisition_id is available,
perform group-disjoint stratified splitting so no group occurs in more
than one partition.

Because exact class stratification and exact group isolation can
conflict, implement an optimization or repeated-search procedure that
minimizes deviation from target class proportions while preserving
group isolation.

Report:

- actual class counts and percentages per partition;
- group counts per partition;
- deviation from requested proportions;
- confirmation that no group leakage exists.

Fallback strategy:

If no grouping column exists, use row-level stratified splitting but
issue a warning that pixel-level random splitting can overestimate
generalization.

Save:

- split_manifest.csv
- split_summary.json
- train_indices.mat
- validation_indices.mat
- test_indices.mat

=======================================================================
7. NORMALIZATION WITHOUT DATA LEAKAGE
=======================================================================

Apply feature-wise min–max normalization:

x'_j = (x_j - min_j) / (max_j - min_j)

Critical requirement:

Compute min_j and max_j from the training partition only.

Use the same training-derived parameters for:

- training
- validation
- testing
- MATLAB inference
- ONNX inference
- quantized ONNX inference

Handle constant training features safely:

- if max_j = min_j, map the feature to zero;
- record the feature as constant;
- issue a warning.

For validation or test values outside the training range, support:

OutOfRangePolicy = "clip" | "allow" | "error"

Default:

OutOfRangePolicy = "clip"

When clipping occurs, record the count and percentage of clipped values
for every feature and partition.

Save preprocessing parameters in:

- preprocessing.mat
- preprocessing.json

The metadata must contain:

- feature order
- class order
- training minima
- training maxima
- constant-feature flags
- hue convention
- delta mode
- clipping policy
- healthy delta references, when applicable
- MATLAB release
- random seed
- training timestamp
- dataset checksum

=======================================================================
8. NETWORK ARCHITECTURE
=======================================================================

Implement the following multilayer perceptron:

Input(7)
→ Dense(64)
→ ReLU
→ Dropout(0.15)
→ Dense(32)
→ ReLU
→ Dropout(0.15)
→ Dense(16)
→ ReLU
→ Dense(5)
→ Softmax

Use featureInputLayer and do not normalize again inside the network,
because external min–max preprocessing is required for portability.

Suggested MATLAB architecture:

featureInputLayer(numFeatures, ...
    Normalization="none", ...
    Name="features")

fullyConnectedLayer(64, ...
    WeightsInitializer="he", ...
    BiasInitializer="zeros", ...
    Name="fc64")

reluLayer(Name="relu64")
dropoutLayer(0.15, Name="dropout64")

fullyConnectedLayer(32, ...
    WeightsInitializer="he", ...
    BiasInitializer="zeros", ...
    Name="fc32")

reluLayer(Name="relu32")
dropoutLayer(0.15, Name="dropout32")

fullyConnectedLayer(16, ...
    WeightsInitializer="he", ...
    BiasInitializer="zeros", ...
    Name="fc16")

reluLayer(Name="relu16")

fullyConnectedLayer(5, ...
    WeightsInitializer="he", ...
    BiasInitializer="zeros", ...
    Name="class_logits")

softmaxLayer(Name="probabilities")

For DeltaMode="exclude", automatically change numFeatures from 7 to 6
while keeping the remaining architecture unchanged.

Initialize all dense-layer weights explicitly using He initialization.
Verify the initialized network with analyzeNetwork or an equivalent
programmatic check before training.

Report:

- number of trainable parameters;
- approximate FP32 parameter memory;
- input and output tensor layouts.

=======================================================================
9. CLASS-IMBALANCE HANDLING
=======================================================================

Calculate class weights from the training partition only.

Default weighting method:

w_k = N / (K n_k)

where:

- N is the number of training observations;
- K=5 is the number of classes;
- n_k is the training count of class k.

Normalize the weights so their mean is 1.

Also support:

ClassWeightMethod =
    "none" |
    "inverse-frequency" |
    "effective-number"

For effective-number weighting, expose beta as a configurable parameter.

Implement weighted categorical cross-entropy using trainnet and a
custom loss function compatible with automatic differentiation.

For one-hot targets:

L = -(1/B) Σ_i Σ_k w_k y_ik log(p_ik + epsilon)

Use a numerically stable epsilon value.

Do not apply class weights to validation or test metrics.

Print and save:

- training class frequencies;
- raw class weights;
- normalized class weights;
- selected weighting method.

=======================================================================
10. TRAINING CONFIGURATION
=======================================================================

Use Adam optimization with defaults:

- InitialLearnRate = 0.001
- MaxEpochs = 200
- MiniBatchSize = configurable, default 4096
- GradientThreshold = configurable
- L2Regularization = configurable, default 1e-4
- Shuffle = "every-epoch"
- ExecutionEnvironment = "auto"

Use validation data throughout training.

Implement early stopping with:

- ValidationPatience = configurable, default 10 validation evaluations;
- OutputNetwork = "best-validation";
- ObjectiveMetricName = "loss", unless a properly implemented custom
  macro-F1 metric is explicitly selected.

Choose ValidationFrequency so validation occurs at least once per epoch.
Clearly state that MATLAB patience counts validation checks, not
necessarily epochs.

Record:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- learning rate;
- epoch;
- iteration;
- elapsed time;
- stopping reason.

Save the complete training history to:

- training_history.csv
- training_history.mat
- training_curves.png

Save checkpoints in a configurable checkpoint directory.

The code must return the network corresponding to the best validation
objective, not merely the final iteration.

=======================================================================
11. MODEL EVALUATION
=======================================================================

Evaluate the final selected model independently on:

- training set
- validation set
- held-out test set

Compute:

1. Overall accuracy
2. Balanced accuracy
3. Macro precision
4. Macro recall
5. Macro F1-score
6. Weighted F1-score
7. Per-class precision
8. Per-class recall or sensitivity
9. Per-class specificity
10. Per-class F1-score
11. Per-class support
12. Confusion matrix

Use safe zero-division handling.

Confusion-matrix convention must be:

- rows = true classes
- columns = predicted classes

Use class order:

H, C1, C2, C3, C4

Generate both:

- count confusion matrix;
- row-normalized confusion matrix.

Export:

- metrics_train.json
- metrics_validation.json
- metrics_test.json
- per_class_metrics_test.csv
- confusion_matrix_counts.csv
- confusion_matrix_normalized.csv
- confusion_matrix_counts.png
- confusion_matrix_normalized.png

For PhD-level reporting, calculate seeded bootstrap 95% confidence
intervals for at least:

- test accuracy;
- test balanced accuracy;
- test macro-F1.

Use group-level bootstrap resampling when group identifiers are
available. Otherwise, use a clearly labeled observation-level bootstrap.

Make the number of bootstrap replicates configurable, default 1000.

=======================================================================
12. ADDITIONAL RESEARCH DIAGNOSTICS
=======================================================================

Generate:

- class-distribution plots;
- feature distributions by class;
- feature correlation matrix;
- normalized feature distributions;
- delta leakage diagnostics;
- predicted-probability distributions;
- model confidence histograms;
- reliability diagram;
- expected calibration error, if practical;
- misclassification table containing true label, predicted label,
  confidence, feature values, and source metadata.

If both seven-feature and six-feature models are trained, generate a
comparison table containing:

- accuracy;
- balanced accuracy;
- macro-F1;
- per-class F1;
- inference time;
- model size;
- delta-related leakage diagnostics.

Do not choose the seven-feature model automatically. Select the primary
model using validation macro-F1, subject to the leakage analysis.

=======================================================================
13. MODEL SERIALIZATION
=======================================================================

Save the MATLAB artifacts:

- corrosion_model.mat
- preprocessing.mat
- class_names.mat
- training_configuration.json
- experiment_summary.json

The MAT file should contain:

- trained dlnetwork;
- feature names;
- class names;
- preprocessing parameters;
- selected model type;
- delta configuration;
- class weights;
- training history;
- evaluation results;
- software version information.

Use versioned experiment directories so previous runs are never
silently overwritten.

Example:

outputs/experiments/2026-06-27_143000_seed42/

=======================================================================
14. FP32 ONNX EXPORT
=======================================================================

Export the selected network to:

corrosion_model_fp32.onnx

Requirements:

- dynamic batch dimension;
- input feature count fixed at 7 or 6 according to the selected model;
- output dimension fixed at 5;
- meaningful input and output node names;
- configurable supported ONNX opset;
- no placeholder or unsupported custom operators.

Use dynamic batching explicitly, for example:

exportONNXNetwork(net, onnxPath, ...
    BatchSize=[], ...
    NetworkName="CopperCorrosionMLP", ...
    OpsetVersion=<supported value>)

Run analyzeNetwork before export.

After export:

1. Load the ONNX model with ONNX Runtime.
2. Run predictions on a deterministic subset of test observations.
3. Compare MATLAB and ONNX probabilities.
4. Report:

   - maximum absolute probability difference;
   - mean absolute probability difference;
   - top-1 prediction agreement;
   - ONNX test metrics.

Use configurable tolerances, such as:

- MaxProbabilityErrorTolerance = 1e-5
- MinimumPredictionAgreement = 0.999

Fail verification with a clear diagnostic if tolerances are exceeded.

=======================================================================
15. QUANTIZED ONNX EXPORT
=======================================================================

Do not falsely claim that a MATLAB quantized dlnetwork can always be
exported directly as a portable INT8 ONNX model.

Implement a reliable ONNX Runtime quantization stage.

Generate:

python/quantize_onnx.py

The MATLAB project should invoke the script using either:

- pyrunfile, or
- a system command using a configured Python executable.

Default output:

corrosion_model_int8.onnx

For this dense MLP, implement dynamic INT8 quantization of supported
Gemm or MatMul operators using ONNX Runtime.

Also support an optional static QDQ quantization mode using a
representative calibration subset drawn from the training partition
only.

QuantizationMode = "dynamic" | "static"

Never calibrate quantization using validation or test data.

The quantization script must:

- preprocess or perform ONNX shape inference when required;
- preserve dynamic batch support when supported;
- quantize supported weights/operators;
- save a valid ONNX model;
- print the operators that were quantized;
- report original and quantized file sizes;
- return a nonzero exit code on failure.

Validate the quantized model on the held-out test set.

Compare FP32 ONNX and INT8 ONNX using:

- accuracy;
- balanced accuracy;
- macro-F1;
- per-class F1;
- top-1 agreement;
- mean absolute probability difference;
- model file size;
- inference latency;
- throughput.

Use a configurable acceptance criterion, for example:

MaximumMacroF1Drop = 0.01

If the quantized model exceeds the allowed degradation, keep the model
but mark it as not accepted for deployment.

Optionally provide a native MATLAB INT8 experiment using dlquantizer
when the required add-on is installed, but clearly distinguish this
artifact from the portable INT8 ONNX model.

=======================================================================
16. IMAGE FEATURE EXTRACTION FOR INFERENCE
=======================================================================

Implement image inference for RGB images.

Input formats should include common MATLAB-supported image formats such
as PNG, JPEG, TIFF, and BMP.

For each input image:

1. Read the image.
2. Remove or handle an alpha channel if present.
3. Convert grayscale images to RGB or reject them according to config.
4. Convert integer RGB values consistently to floating point.
5. Compute CIELAB values using MATLAB color conversion.
6. Compute HSV values using rgb2hsv.
7. Ensure H, S, and V use the same convention as training.
8. Compute or load delta according to DeltaMode.
9. Arrange features in exactly this order:

   [L, a, b, H, S, V, delta]

10. Apply the stored training-derived min–max scaler.
11. Process the pixels in configurable chunks.
12. Reconstruct the prediction map in the original image dimensions.

Never estimate new normalization parameters from an inference image.

For DeltaMode="provided", support:

- a precomputed delta image;
- a MAT file containing a delta map;
- a tabular per-pixel feature input containing x and y coordinates.

For DeltaMode="ciede2000", use the healthy references stored in the
training metadata.

If the model requires delta but no valid delta source or healthy
reference is available, stop with a clear error. Do not silently use
zero for every inference pixel.

=======================================================================
17. BATCHED INFERENCE
=======================================================================

Implement configurable chunked prediction:

ChunkSize = configurable, default 100000 pixels

The code must:

- avoid loading unnecessary duplicate feature matrices;
- preallocate outputs;
- support CPU and GPU where available;
- preserve output ordering;
- handle the final partial chunk;
- return probabilities optionally;
- display progress only when requested.

Support:

ReturnProbabilities = true | false

For large images, add an optional memory-conscious mode that computes
and classifies image rows or tiles incrementally.

=======================================================================
18. SEGMENTATION OUTPUTS
=======================================================================

Use class IDs:

- H  = 0
- C1 = 1
- C2 = 2
- C3 = 3
- C4 = 4

Use this exact visualization legend:

- H  = transparent
- C1 = RGB(255, 0, 0)
- C2 = RGB(0, 0, 0)
- C3 = RGB(209, 225, 225)
- C4 = RGB(70, 174, 194)

Generate:

1. labels.png

   A single-channel uint8 class-index image containing values 0–4.

2. mask.png

   A colorized PNG with alpha transparency:
   - H pixels must be fully transparent;
   - corrosion classes must use the exact colors above.

3. overlay.png

   The original RGB image blended with the corrosion colors.
   Healthy pixels must remain unchanged.

Make corrosion overlay opacity configurable:

OverlayAlpha = 0.50

4. probabilities.mat, optional

   Per-pixel class probabilities, preferably saved only when explicitly
   requested because of memory requirements.

5. legend.png

   A small generated legend containing class names, chemical names, and
   colors.

Do not use lossy image formats for masks.

=======================================================================
19. POST-PROCESSING
=======================================================================

Provide optional, disabled-by-default connected-component filtering.

Configuration:

EnableSmallRegionRemoval = false
MinimumRegionSize = 5
Connectivity = 8

When enabled:

- process every corrosion class separately;
- remove connected components smaller than MinimumRegionSize;
- assign removed pixels to H;
- record how many components and pixels were removed from each class.

Save both raw and post-processed outputs when post-processing is enabled.

Do not apply morphological filtering silently.

=======================================================================
20. INFERENCE STATISTICS
=======================================================================

Save stats.json containing at least:

- source image filename;
- image width and height;
- total pixels;
- model feature count;
- model version;
- delta mode;
- chunk size;
- execution environment;
- class counts;
- class percentages of all pixels;
- class percentages among corrosion pixels;
- total healthy pixels;
- total corrosion pixels;
- corrosion coverage percentage;
- preprocessing time;
- feature extraction time;
- normalization time;
- prediction time;
- post-processing time;
- output-writing time;
- total runtime;
- pixels per second;
- peak or estimated memory usage, when practical;
- MATLAB release;
- CPU/GPU information;
- timestamp.

Use valid JSON types and avoid encoding numeric values as strings.

=======================================================================
21. BENCHMARK MODE
=======================================================================

Implement benchmark mode for:

- MATLAB FP32 model;
- FP32 ONNX model;
- INT8 ONNX model;
- optional native MATLAB quantized model.

Configuration should include:

BenchmarkWarmupRuns = 3
BenchmarkMeasuredRuns = 20
BenchmarkChunkSizes = [1000, 10000, 100000, 500000]
BenchmarkBatchSizes = configurable

Benchmark separately:

1. model-only inference;
2. preprocessing plus inference;
3. complete end-to-end image processing including output reconstruction.

Use warm-up runs that are excluded from reported statistics.

Report:

- mean latency;
- median latency;
- standard deviation;
- minimum;
- maximum;
- 95th percentile;
- pixels per second;
- samples per second;
- model load time;
- model file size;
- estimated or measured memory use;
- prediction agreement with MATLAB FP32.

Save:

- benchmark_results.csv
- benchmark_results.json
- benchmark_latency.png
- benchmark_throughput.png
- benchmark_model_size.png

Do not mix disk I/O time with model-only inference time.

Synchronize GPU execution before stopping timers where necessary.

=======================================================================
22. REPRODUCIBILITY AND EXPERIMENT TRACKING
=======================================================================

Every experiment must save:

- random seed;
- configuration;
- input filename;
- input-file checksum;
- row count;
- class distribution;
- split manifest;
- preprocessing parameters;
- architecture;
- parameter count;
- training history;
- best validation iteration;
- stopping reason;
- metrics;
- MATLAB version;
- toolbox versions;
- hardware information;
- ONNX opset;
- Python version;
- ONNX Runtime version;
- quantization method;
- output checksums.

Generate a human-readable experiment report in Markdown:

experiment_report.md

The report should summarize:

- dataset;
- leakage checks;
- split methodology;
- preprocessing;
- model;
- training;
- test results;
- ablation findings;
- ONNX verification;
- quantization impact;
- benchmarking;
- limitations.

=======================================================================
23. SOFTWARE QUALITY
=======================================================================

The implementation must:

- use functions rather than one monolithic script;
- use inputParser or an equivalent validated configuration mechanism;
- validate all public function arguments;
- use descriptive variable names;
- use preallocation;
- avoid unnecessary loops when vectorization is clear;
- avoid hard-coded file paths;
- use fullfile for paths;
- create missing output directories;
- use try/catch only where it adds meaningful context;
- rethrow errors when the program cannot safely continue;
- use comments explaining scientific or non-obvious implementation
  decisions;
- include MATLAB help blocks for major functions;
- avoid global variables;
- avoid dependence on workspace state;
- support Windows and Linux path conventions;
- save figures without requiring interactive display;
- support headless execution where practical.

=======================================================================
24. UNIT AND INTEGRATION TESTS
=======================================================================

Generate executable matlab.unittest tests.

Tests must cover:

- CSV loading;
- Parquet loading when supported;
- missing columns;
- invalid labels;
- NaN and Inf detection;
- hue-convention detection;
- range validation;
- conflicting duplicate detection;
- exact class-wise split accounting;
- group leakage prevention;
- reproducible splitting;
- training-only scaler fitting;
- constant-feature normalization;
- out-of-range clipping;
- class-weight calculation;
- metric calculation using a known confusion matrix;
- zero-division handling;
- CIEDE2000 reference cases;
- output image dimensions;
- exact legend colors;
- mask transparency;
- chunked versus unchunked prediction equivalence;
- MATLAB versus ONNX prediction agreement;
- stats.json schema;
- benchmark result generation.

Also generate a small synthetic dataset generator so the full pipeline
can be tested without the real research data.

=======================================================================
25. FINAL RESPONSE FORMAT
=======================================================================

Present the response in this order:

1. Critical scientific and technical analysis
2. Corrections made and reasons
3. Assumptions
4. Software prerequisites
5. Project directory tree
6. Installation instructions
7. Complete code for every MATLAB file
8. Complete code for every Python helper file
9. Example configuration
10. Synthetic-data example
11. Training command
12. Image-inference command
13. ONNX export and verification command
14. INT8 quantization command
15. Benchmark command
16. Expected output files
17. Troubleshooting guide
18. Reproducibility and research-validity notes

All code must be internally consistent and immediately runnable after
the documented dependencies are installed.

Do not:

- provide pseudocode;
- omit helper functions;
- invent unavailable MATLAB functions;
- normalize using validation or test data;
- calculate healthy references from validation or test data;
- perform pixel-wise splitting without warning about leakage;
- force delta=0 at inference;
- claim successful INT8 ONNX export without validating the produced model;
- report only accuracy;
- choose a model based on test-set results;
- silently remove observations or image regions;
- silently change the requested class-color legend.

Where a requirement depends on the exact MATLAB release, detect the release programmatically and provide a compatible alternative with a clear explanation.