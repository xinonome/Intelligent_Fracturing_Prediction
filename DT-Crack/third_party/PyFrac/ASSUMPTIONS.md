# V1 assumptions and scientific boundaries

- Each stage is an independent planar 3D PyFrac simulation. No fracture state, width, pressure, or induced stress modification is passed from one stage to another.
- Local coordinates use `u` transverse to the well and `v` positive upward. Input TVD is positive downward; exported global `Z` is positive upward and `Z = -TVD`.
- The default fracture plane is vertical and transverse to the local well tangent. Near-vertical well segments require an explicit `fracture_azimuth_deg`; orientation is never guessed.
- PyFrac 1.1.1 currently requires homogeneous `Eprime` for one simulation. Each stage uses the median of `E/(1-nu^2)` and records the range, standard deviation, and warning when spread is high.
- Toughness and Carter leak-off are read from explicitly named log columns or explicit YAML constants. No values are inferred from GR, E, or empirical guesses.
- The baseline/demo path uses the project adapter's snapshot mode to initialize a PKN state at requested output times. It is a fast model-space workflow, not a replacement for a full native time-marched production run.
- The project adapter currently returns aggregate scalar width/pressure and no independent leak-off volume. The workflow therefore reports mass balance as not computed and never treats a residual as leak-off.
- Outputs and plots are engineering diagnostics. They are not independent evidence of field fracture geometry accuracy.
