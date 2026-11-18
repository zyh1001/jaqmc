# Hall geometries

`jaqmc hall train` and `jaqmc hall evaluate` select the geometry through
`system.module`. Omitting it retains the spherical system and its existing
configuration. Each geometry uses its own wavefunction, kinetic energy,
potential energy and sampling proposal.

```bash
# Disk training
jaqmc hall train system.module=jaqmc.app.hall.config:HallDiskConfig \
  system.nspins='[6,0]' system.flux=18 system.interaction_strength=1 \
  workflow.save_path=runs/disk

# Torus training with fractional electron coordinates
jaqmc hall train system.module=jaqmc.app.hall.config:HallTorusConfig \
  system.nspins='[2,0]' system.flux=6 system.tau='[0.0,1.0]' \
  workflow.save_path=runs/torus
```

These commands run numerical calculations and belong in compute-node jobs.
The integration-test submission instructions are in the repository's
`INTEGRATION.md`.

## Coordinates and energy conventions

| Geometry | Electron coordinates | Hamiltonian |
| --- | --- | --- |
| Sphere | Angles $(\theta,\phi)$ | Upstream spherical magnetic kinetic energy and chord-distance Coulomb energy |
| Torus | Fractional coordinates $(u,v)$ | Symmetric-gauge magnetic kinetic energy and two-dimensional Ewald Coulomb energy |
| Disk | Cartesian coordinates $(x,y)$ | Planar magnetic kinetic energy, electron repulsion and the potential of a positive background disk |

Torus positions obey $z=L_1(u+\tau v)$, with
$L_1^2\operatorname{Im}\tau=2\pi N_\phi$. `system.flux` specifies $N_\phi$,
and `system.tau` is serialized as `[real, imaginary]`. The Ewald energy
includes the neutralizing background and periodic self-image contribution.
`wf.module=jaqmc.app.hall.wavefunction.torus.mhpo` selects the default ansatz.
The alternate `torus.sigma_mhpo` module retains its original restriction to
rectangular tori and its deprecation warning. Its validated case is the square
torus. It does not add support for arbitrary complex `tau`.

Disk positions use the convention $z=x-iy$. The background radius defaults
to $\sqrt{2\,\mathtt{flux}}$, and `system.d` is its vertical separation from
the electrons. This radius sets the background charge distribution and the
initial sample region. Sampling has no hard radial boundary. The reported
disk potential omits the constant background self-energy. Its original
integration and interpolation settings are retained.

The physical interaction scaling is `system.interaction_strength`.
Fitted symbolic disk states retain their separate `wf.kappa` parameter;
both settings must describe the intended physical calculation.

The neural disk orbital count defaults to `system.flux`. Either
`system.num_orbitals` or `wf.num_orbitals` sets the count. Explicit settings
in both locations must agree.

Disk Laughlin initialization requires a positive odd integer
$m=\mathtt{flux}/N$ to preserve fermionic exchange symmetry.
Fitted symbolic states require finite nonnegative `wf.kappa`. Rational
denominators must be nonzero, and the negative quartic term must retain its
confining sign. For `DiskPhi4`, `DiskPhi7`, `DiskPhi8`, `DiskP6`, and
`DiskPhi7RefitC363`, the respective upper bounds are approximately 29.17,
28.94, 28.25, 25.03, and 23.92. The upper endpoint and any internal poles
are excluded. These domain checks preserve the fitted coefficients and do
not establish accuracy throughout the allowed interval.

For `DiskP10C459` and `DiskP11C476`, the combined quartic form
$C R_4+D D_4$ must satisfy $C+24\min(D,0)\geq0$, where
$R_4=\sum_i|z_i|^4$ and $D_4=\sum_{i<j}|z_i-z_j|^4$ for six electrons.
This excludes their growing quartic envelopes at small coupling, including
$\kappa=0$.

## Disk observables

The existing `estimators.enabled` flags select `density`, `pair_correlation`,
`one_rdm`, `structure_factor`, `edge_green`, and `wavefunction_scan`.
`short_range` additionally collects distance-resolved energy sums. It requires
energy and angular momentum, and its `kappa` must equal
`system.interaction_strength`.

```bash
jaqmc hall evaluate system.module=jaqmc.app.hall.config:HallDiskConfig \
  workflow.source_path=runs/disk workflow.save_path=runs/disk-observables \
  estimators.enabled.structure_factor=true estimators.enabled.short_range=true
```

Angular-momentum penalties use complex local operator values during training.
Reported angular-momentum means and variances use the real parts.

`edge_green` uses a separate evaluation with the first electron pinned at
`estimators.edge_green.edge_radius`. The radius defaults to `system.radius`,
or $\sqrt{2\,\mathtt{flux}}$ when no system radius is set.
An explicit `sampler.edge_radius` must agree. Enabling `edge_green` disables
energy and angular-momentum estimation by default. Explicitly enabling energy,
angular momentum, density, pair correlation, one-RDM, structure factor, or
short-range estimation in the same run raises a configuration error. Training
and angular penalties are also incompatible with edge Green-function
estimation. Conditional wavefunction scans may accompany it. Equilibrium
observables use an independent run with the ordinary sampler.
Empty density bins in the disk pair correlation remain undefined and are
reported as `NaN`.

## Wavefunction overlap

Disk, Sphere, and Torus overlaps use the symmetric sampling density
$\rho\propto|\psi|\,|\psi_{\rm ref}|$ through a dedicated command:

```bash
jaqmc hall overlap workflow.source_path=runs/disk-current \
  workflow.reference_path=runs/disk-reference workflow.save_path=runs/disk-overlap \
  system.nspins='[6,0]' system.flux=18
```

Both checkpoints must use the configured wavefunction architecture and parameter
structure. Omitting `workflow.reference_path` performs a self-overlap.
Parameter-free states may also omit `workflow.source_path`. The digest is
`overlap_digest.npz`, retaining `cos_gamma_modulus`, `f_phase`, `cosine`, and
`fubini_distance`.

The default geometry for `hall overlap` is Disk. Sphere and Torus select
`system.module=jaqmc.app.hall.config:HallSphereConfig` and
`system.module=jaqmc.app.hall.config:HallTorusConfig`, respectively. The command
uses each geometry's ordinary proposal, including the sphere area measure and
torus periodic wrapping, with the overlap density in place of the training
density. The estimators are `DiskFubini`, `SphereFubini`, and `TorusFubini` in
the corresponding Hall estimator directories.

For a training checkpoint $\Psi_t$ and the fixed final state $\Psi_f$, all
integrals use the configuration-space measure of the selected geometry. Define

$$
D_{\mathrm{FS}}(t)=\arccos\frac{|\langle\Psi_f|\Psi_t\rangle|}
{\sqrt{\langle\Psi_f|\Psi_f\rangle\langle\Psi_t|\Psi_t\rangle}}.
$$

Sampling $p_t(R)\propto|\Psi_t(R)\Psi_f(R)|$ and writing
$d=\log|\Psi_t|-\log|\Psi_f|$ gives

$$
C_{\mathrm{mod}}=
\left[\langle e^d\rangle_{p_t}\langle e^{-d}\rangle_{p_t}\right]^{-1/2},
\qquad
F_{\mathrm{phase}}=
\left|\left\langle e^{i(\arg\Psi_f-\arg\Psi_t)}\right\rangle_{p_t}\right|,
$$

$$
\cos D_{\mathrm{FS}}=C_{\mathrm{mod}}F_{\mathrm{phase}}.
$$

Both amplitude expectations use log-sum-exp across walkers, devices, and
sampling steps. The phase is averaged as a complex quantity before taking its
magnitude. Overall amplitude and phase factors cancel. Each pair requires its
own overlap-distribution sampling run. Averaging stepwise distances does not
give the distance computed from the accumulated overlap statistics.

`estimator/fubini.py` contains the shared Hall implementation. The geometry
modules preserve the disk API and expose the sphere and torus estimators.
The existing `DiskOverlapConfig`, `DiskOverlapWorkStage`, and disk
`FubiniParams` import paths remain supported.

### Checkpoint-distance series

The script `scripts/hall_fubini.py` prepares one native overlap evaluation for
every saved checkpoint in the selected source dataset. Its final reference
steps are 149999 for Disk, 199999 for Sphere, and 1999999 for Torus. Checkpoint
selection uses saved steps rather than interpolating unavailable wavefunctions.

```bash
python scripts/hall_fubini.py prepare sphere --data-root /path/to/data \
  --output runs/fubini/sphere --batch-size 4096 --steps 10000 --burn-in 1000
python scripts/hall_fubini.py run --output runs/fubini/sphere --dry-run
python scripts/hall_fubini.py run --output runs/fubini/sphere
```

`--checkpoint-steps 1999 199999` selects a subset during preparation. Input
copies preserve all parameter arrays and select walkers reproducibly without
replacement. The plan records source and prepared-file checksums, the final
reference, and per-checkpoint configurations. Each evaluation saves its log,
step statistics, checkpoint, and digest in a separate directory. Re-running
`run` resumes unfinished evaluations and retains existing completed digests.
Collection validates the requested step counts and overlap factorization,
then writes `fubini_series.npz` and `fubini_summary.json`. Collection can also
run separately with `hall_fubini.py collect --output ...`.

Distances are in radians. The series does not currently include standard
errors. The output must be imported into the merged dataset before publication
plotting. `plot_fubini.py` reads only the merged `fubini/step` and
`fubini/distance` fields.

`scripts/evaluate_hall_fubini.sh` runs one prepared geometry on the `gpu`
partition with two GPUs and eight CPU cores. From the cluster project root:

```bash
sbatch --export=ALL,RUN_ID=hall_fubini_20261115 scripts/evaluate_hall_fubini.sh disk
sbatch --export=ALL,RUN_ID=hall_fubini_20261115 scripts/evaluate_hall_fubini.sh sphere
sbatch --export=ALL,RUN_ID=hall_fubini_20261115 scripts/evaluate_hall_fubini.sh torus
```

The script accepts `PROJECT_ROOT`, `DATA_ROOT`, and `RESULT_ROOT` overrides.
The default result root is `/home/jqguo/project/yuhang/runs`. A new run ID
creates a new plan. An existing plan resumes with its recorded settings.
Additional preparation options, such as `--checkpoint-steps`, follow the
geometry argument when creating a plan.

`scripts/submit_hall_fubini.sh` submits all three geometries and records the
job identifiers in `<RESULT_ROOT>/<RUN_ID>/jobs.tsv`. It retains existing job
identifiers when invoked again with the same run ID.

```bash
RUN_ID=hall_fubini_20261116_r1 bash scripts/submit_hall_fubini.sh
```

On the local host, the collection script can wait for all three summaries,
download the complete run with `rsync`, validate the reference checkpoints and
step statistics, back up the existing merged dataset, import the overlap
series, and generate the four distance figures from merge:

```bash
python scripts/collect_hall_fubini.py \
  --remote paracloud \
  --remote-root /home/jqguo/project/yuhang/runs/hall_fubini_20261116_r1 \
  --run-root runs/hall_fubini_20261116_r1 \
  --merge-root /home/ubuntu/work/data/merge \
  --figures-root /home/ubuntu/work/data/figures \
  --plot-python /home/ubuntu/work/data/figures/.venv/bin/python --watch
```

Without `--watch`, the script checks once and returns if sampling is incomplete.
The collection command runs in the JaQMC environment. The separate plotting
interpreter defaults to `<figures-root>/.venv/bin/python`.
Its progress is recorded in `collection_status.json`. Failed scheduler jobs
stop collection before importing any data. The standalone importer is
`scripts/merge_hall_fubini.py --run-root ... --merge-root ...`.
It adds `fubini/step`, `distance`, overlap factors, the reference identity,
sampling settings, and the three step-statistic arrays under `fubini/series/`.
Each checkpoint can override `iterations` and `burn_in` in `fubini_plan.json`.
The corresponding `run` settings in its `requested_config.yaml` must agree.
Completed checkpoints retain their original settings when unfinished comparisons
are restarted with shorter sampling. The merged `iterations_per_checkpoint`,
`burn_in_per_checkpoint`, and `total_configurations_per_checkpoint` arrays record
the actual settings for every checkpoint. Step-statistic rows are padded with
`NaN` beyond their valid lengths given by `iterations_per_checkpoint`.
`manifest.json`, `schema.json`, and `fubini_validation.json` describe the sources
and checks. Existing observables and energy references are retained.

Torus checkpoint comparisons retain their importance-sampling estimator:

```bash
jaqmc hall evaluate system.module=jaqmc.app.hall.config:HallTorusConfig \
  workflow.source_path=runs/torus workflow.save_path=runs/torus-overlap \
  estimators.enabled.fubini=true
```

It compares the training checkpoints in the source directory against the last
saved step by default. `estimators.fubini.checkpoint_path` and
`estimators.fubini.reference_step` select another directory and reference step.
The result uses the original `fubini:*` keys.
Checkpoint and reference amplitudes use independent numerical scales so
arbitrary global normalization factors cancel from the overlap and distance.

## Checkpoints and validation

The upstream checkpoint format and Sphere parameter trees are retained. Disk
parameter names and envelope modes are retained, including `amplitude_imag`
and `m_learnable2_old`. A checkpoint created before an extra parameter existed
still needs an explicit parameter conversion. No parameters are silently filled.

Histogram and wavefunction-scan evaluation states are adapted to the current
upstream device-state layout. Start fresh evaluation output directories when
using a migrated training checkpoint. Old accumulated evaluation states are
not converted.
Torus pair histograms use replicated two-dimensional state. Their
normalization remains $N^2$, with an uncorrelated mean of $(N-1)/N$.

Source comparisons load saved original modules under the current upstream
framework and compare identical coordinates and parameters. They cover the
migrated mathematical implementations, derivatives and local energies.
They do not establish equivalence between two historical training environments.
All numerical tests remain pending until submitted on compute nodes.
