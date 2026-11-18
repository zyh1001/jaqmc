# Quantum Hall Training

Configuration reference for `jaqmc hall train`.
This page shows the effective defaults for the train workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Keys use the same
dot notation as CLI overrides, such as `train.run.iterations=5000`. Defaults
are resolved in this order: schema defaults, workflow preset, YAML config, then
CLI overrides. For evaluation config, see <project:eval.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## ParaCloud sphere and disk retraining

From the local project directory, the following commands synchronize the project,
submit two independent jobs, inspect their status, and download completed results:

```bash
bash scripts/retrain_final.sh submit
bash scripts/retrain_final.sh status <run_id>
bash scripts/retrain_final.sh download <run_id>
```

The submission command prints the run identifier. An optional identifier after
`submit` reuses that submission record and submits only geometries without a
recorded job. Failed or interrupted training is reported without automatic
resubmission. Submission and download are separate operations.

A short GPU memory validation preserves the total batch size and optimizer
settings while evaluating walkers in chunks:

```bash
bash scripts/retrain_final.sh submit <run_id> --steps 20 --chunk-size 128
```

`--chunk-size` applies to kinetic energy, loss gradients, and the sphere's
angular momentum estimator. `--steps` overrides only the training iteration
target. The completion check follows the configured target, so a twenty-step
validation requires `train_ckpt_000019.npz`. Without these options, the
historical iteration targets and unchunked evaluation remain unchanged.

`scripts/validate_final_memory.py <run_id>` monitors this twenty-step validation.
It verifies twenty finite energy and variance records plus the final checkpoint.
Only geometries with a confirmed out-of-memory failure proceed to an eight-GPU
validation. Other failures are reported without changing the training settings.
Progress and the final result are written to
`../data/final/<run_id>/submission/validation_report.json`.

Eight-GPU jobs use `--gpus 8`, two nodes with four GPUs each, one process and
six CPU cores per node. The total batch size remains 8192. Both processes
initialize JAX together and verify eight global GPUs and four local GPUs.
`--geometry sphere` or `--geometry disk` selects one geometry, and
`--time-limit 01:00:00` bounds a short validation. Default submissions still use
four GPUs, twelve CPU cores, both geometries, and no time limit.

Both jobs request four GPUs and twelve CPU cores in the `gpu` partition, with
`--time=0`. Training starts from fresh parameters with four network layers,
`iqr` clipping, KFAC, and 200 learning-rate warmup steps. The sphere runs for
200,000 steps and the disk for 150,000 steps. Other historical settings are
preserved, with field mappings for the current sphere model and disk config.
The source configurations remain unchanged.

The local entry point delegates configuration and transfer operations to
`scripts/retrain_final.py`. The source configurations are
`../data/spherical/mhpo-1det/train/train_config.yaml` and
`../data/disk/n6_Q18_kapa1_8192_final/train_config.yaml`. Code is synchronized to
`paracloud:/home/jqguo/project/jaqmc_merge`, excluding environments, caches,
historical outputs, and logs. Replaced remote files are backed up without
deleting destination files. The existing cluster environment is reused.

Results are stored under
`/home/jqguo/project/yuhang/runs/<run_id>/{sphere,disk}` and downloaded to
`../data/final/<run_id>`. The run also contains logs, original and prepared
configurations, configuration differences, source checksums, and job identifiers.
Download checks both the Slurm exit status and the final checkpoint, then
verifies its size and SHA-256 checksum. Incomplete geometries are skipped with
a nonzero exit status while completed geometries can still be downloaded.

`PARACLOUD_HOST`, `REMOTE_PROJECT`, `RESULT_ROOT`, and `LOCAL_RESULT_ROOT`
override the connection alias and destination roots. The source checkout is
located relative to the entry point. Job scripts support Slurm's copied-script
execution through `PROJECT_ROOT` and the explicitly supplied configuration.

```{eval-rst}
.. config-context::
   :preset: jaqmc.app.hall.workflow.HallTrainWorkflow.default_preset
```

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.base.WorkflowConfig
   :prefix: workflow
```

(hall-train-system)=
## System (`system.*`)

Defines the quantum Hall system on the Haldane sphere.

See <project:index.md> for physics background and usage examples.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.config.HallSphereConfig
   :prefix: system
```

(hall-train-wf)=
## Wavefunction (`wf.*`)

Selects and configures the neural network ansatz.

- Default module selection: `mhpo`. Effective defaults for MHPO are
  listed below.

Analytic `laughlin` and `free` benchmarks are evaluation-only; see
<project:eval.md>.

See <project:index.md> for background on each architecture.

### MHPO options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.sphere.mhpo.SphereMHPO
   :prefix: wf
   :scope: MHPO
```

(hall-train-sampler)=
## Sampler (`sampler.*`)

The Hall workflow uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: sampler
```

(hall-train-stage)=
## Train Stage (`train.*`)

The VMC optimization loop. Samples electron configurations on the Haldane sphere, computes energy (and optional angular momentum penalties), and updates wavefunction parameters.

(hall-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: train.run
```

(hall-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for the built-in
  optimizers are listed below.

#### KFAC options

Sphere, disk, and torus MHPO Jastrow parameters are explicitly registered for
first-order updates. Their gradients bypass KFAC preconditioning and remain
subject to the learning rate and global norm constraint. Disk registration
includes both the range parameters and the trainable cusp coefficients.

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
   :prefix: train.optim
   :scope: KFAC
```

#### SR options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.sr.SROptimizer
   :prefix: train.optim
   :scope: SR
```

#### Adam options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.adam
   :prefix: train.optim
   :scope: Adam
```

#### LAMB options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.lamb
   :prefix: train.optim
   :scope: LAMB
```

(hall-train-writers)=
### Writers (`train.writers.*`)

The train stage enables `console`, `csv`, and `hdf5` writers by default.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: train.writers.console
```

#### CSV writer (`train.writers.csv.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.csv.CSVWriter
   :prefix: train.writers.csv
```

#### HDF5 writer (`train.writers.hdf5.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.hdf5.HDF5Writer
   :prefix: train.writers.hdf5
```

### Loss gradients (`train.grads.*`)

The workflow resolves {py:obj}`~jaqmc.estimator.loss_grad.LossAndGrad`
from `train.grads.*`. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the clipping formulas.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.loss_grad.LossAndGrad
   :prefix: train.grads
```

---

(hall-train-estimators)=
## Estimators (`estimators.*`)

Energy estimators are configured programmatically by the workflow and are not
typically overridden via config. The same definitions are used by
<project:eval.md>. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

`TotalEnergy` automatically sums all `energy:`-prefixed components.
`SphericalAngularMomentum` is enabled by default; set
`estimators.enabled.angular_momentum=false` to omit it. When
`system.lz_penalty` or `system.l2_penalty` are nonzero, angular momentum
is required by `PenalizedLoss`. `TotalEnergy` and `PenalizedLoss` are not
configurable via config keys.

### Kinetic energy (`estimators.energy.kinetic.*`)

Covariant kinetic energy on the Haldane sphere. See
[Kinetic energy](../../guide/estimators/kinetic.md#spherical-kinetic-energy).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Angular momentum (`estimators.angular_momentum.*`)

Computes `angular_momentum_z`, `angular_momentum_z_square`, and
`angular_momentum_square` for the Haldane sphere. See
[Angular momentum](../../guide/estimators/angular-momentum.md)
for its formulation.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.angular_momentum.SphericalAngularMomentum
   :prefix: estimators.angular_momentum
```

### Coulomb potential (`estimators.energy.potential.*`)

Coulomb repulsion on the Haldane sphere.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.hamiltonian.SpherePotential
   :prefix: estimators.energy.potential
```

### Continue with new KFAC curvature state

Set `train.run.reset_kfac_state_on_restore=true` to restore an explicit training
checkpoint's parameters, walkers, sampler state and learning-rate state while
initializing new KFAC curvature state. This requires KFAC and an explicit
checkpoint file. Parameter and learning-rate arrays must match the current
model in name, shape and precision. The default is `false`, which restores the
complete optimizer state as usual. The iteration target remains the total step
count, including steps in the source checkpoint.

The disk resume wrapper checks the resolved configuration without submitting a
job and accepts additional configuration arguments for training:

```bash
bash scripts/resume_disk.sh --dry-run
sbatch scripts/resume_disk.sh
```

The wrapper writes to
`runs/n6_Q18_kapa1_8192_final_resume_first_order_2000_<job_id>` under the project
root and resumes from the specified checkpoint at step 150000. It trains
through step 151999 using four V100 GPUs.

Run these commands from the project root. The source run defaults to
`../data/disk/n6_Q18_kapa1_8192_final` relative to that root. `DATA_ROOT` selects
a different data directory, and `SOURCE_ROOT` selects a specific source run.
