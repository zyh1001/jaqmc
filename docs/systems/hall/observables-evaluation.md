# 三种几何的固定参数评估

本流程读取指定训练的最终检查点，在参数保持固定的情况下补充物理量和统计误差。每个几何提交一个作业，使用两张显卡、8 个处理器核心和合计 4096 个行走者。三个作业可同时排队和运行。

## 运行与结果

在集群项目目录运行：

```bash
PROJECT_ROOT=/home/jqguo/project/jaqmc_merge \
RESULT_ROOT=/home/jqguo/project/yuhang/runs \
RUN_ID=hall_observables_20261115 \
bash scripts/submit_hall_observables.sh
```

每个作业依次进行训练态和朗福林态评估。两个阶段各自热化 1000 步、正式采样 10000 步，每步更新采样器 10 次。训练态的能量、结构因子和适用的角动量、密度矩阵来自同一批构型。朗福林态另行采样其自身的概率分布。

每个阶段开始前先做短程运行，检查实际行走者数量、参数恢复、结果有限性和能量求和。短程数据单独保存，不计入正式结果。

结果目录为 `RESULT_ROOT/RUN_ID/geometry/`：

- `inputs/` 保存参数完全相同的评估检查点和行走者选择索引。
- `trained/` 和 `laughlin/` 保存配置、逐步记录、检查点和评估汇总。
- `statistics.npz` 保存分块均值、标准误差、能级占据和结构因子误差。
- `summary.json` 保存便于阅读的能量、角动量和占据数汇总。
- `provenance.json`、`validation.json` 和 `run_manifest.json` 记录输入校验和及验收结果。
- 正式阶段全部成功后才写入 `completion.json`。

训练检查点包含 8192 个行走者。本流程按固定种子无放回选取 4096 个，仅生成评估副本，逐个核对所有参数数组与训练检查点相同。球面旧配置的零磁通附着设置转换为当前等价的 `dressing.degree=0`。

球面的 `retain_unused_spin_parameters=true` 保留旧检查点中的空自旋通道参数，该参数不影响波函数数值。环面的 `include_center_of_mass=false` 保留该训练检查点原有的四个位置特征。默认的新模型行为保持原有设置，评估时显式选择与检查点一致的配置。

本机的 `scripts/collect_hall_observables.py` 可以持续检查三个作业。完成后用 `rsync` 下载结果，核对服务器生成的全部文件校验和，再备份并更新合并数据集。作业失败时停止等待，保留状态供检查。单独导入已下载结果的入口为 `scripts/merge_hall_evaluation.py --run-root RUN_ROOT --merge-root MERGE_ROOT`。原始训练数据和已有环面密度矩阵保留原值。球面对关联通过下述独立入口重新计算。

## 结构因子

圆盘与环面采用粒子密度模式：

$$
\rho_{\mathbf q}=\sum_j e^{-i\mathbf q\cdot\mathbf r_j},\qquad
S(\mathbf q)=\frac{\langle|\rho_{\mathbf q}|^2\rangle}{N},\qquad
S_{\mathrm c}(\mathbf q)=\frac{\langle|\rho_{\mathbf q}|^2\rangle-|\langle\rho_{\mathbf q}\rangle|^2}{N}.
$$

圆盘默认计算 $q\ell_B=0$ 到 $5$ 的 101 个点，每个大小取 8 个方向。先逐方向计算，再平均。环面只取倒格子波矢：

$$
\mathbf q_{hk}=\frac{2\pi}{L_1}\left(h,\frac{k-h\operatorname{Re}\tau}{\operatorname{Im}\tau}\right),\qquad |\mathbf q_{hk}|\ell_B\leq5.
$$

环面保存全部整数标签、波矢和等模长平均值，误差统计保留不同模式间的相关性。

球面采用普通球谐函数：

$$
\rho_{\ell m}=\sum_jY_{\ell m}(\Omega_j),\qquad
S_\ell=\frac{4\pi}{N(2\ell+1)}\sum_m\langle|\rho_{\ell m}|^2\rangle
=1+\frac{2}{N}\left\langle\sum_{i<j}P_\ell(\cos\theta_{ij})\right\rangle.
$$

计算 $\ell=0,\ldots,20$，另保存 $q_\ell=\sqrt{\ell(\ell+1)}/R$。实现只显式保存非负 $m$，利用球谐函数的共轭关系对正 $m$ 赋予两倍权重。每个模式扣除其平均密度的平方后得到连通部分。球面定义参考[球面结构因子研究](https://arxiv.org/html/2410.00165v2)。

完整结构因子满足零模式等于 $N$，连通部分的零模式为零。连通部分采用分块删一法修正有限样本偏差并估计误差。文件同时保存直接代入均值的结果和修正后的结果，不将统计涨落产生的小负值强制改为零。

## 密度矩阵与朗道能级占据

设 $R_a'$ 表示将电子 $a$ 换为辅助坐标，辅助坐标的概率密度为 $p$：

$$
\gamma_{\alpha\beta}=\left\langle\sum_a
\frac{\Psi(R_a')}{\Psi(R)}
\frac{\phi_\alpha(\mathbf r_a)\phi_\beta^*(\mathbf r')}{p(\mathbf r')}\right\rangle,
\qquad N_n=\sum_{\alpha\in n}\operatorname{Re}\gamma_{\alpha\alpha}.
$$

圆盘保留三个朗道能级，每层 20 个轨道，总矩阵大小为 $60\times60$。标签为 $n=0,1,2$，$q=0,\ldots,19$，$m=q-n$，物理角动量为 $-m\hbar$。辅助点在半径 $15\ell_B$ 的圆盘内均匀采样，权重为该圆盘面积。输入记录包含各轨道在积分范围外的最大归一化尾部。

球面采用 $Y_{Q,Q+n,m}$，$Q=7.5$，三个能级轨道数分别为 16、18、20，总矩阵大小为 $54\times54$。辅助点在单位球面均匀采样，角度积分权重为 $4\pi$。基函数只累加非零项，使用非负整数次幂，避免两极处出现不定值。

保存原始复矩阵、对角占据、能级总占据及矩阵迹。能级占据的误差由每一步的能级总和计算。不计算自然轨道占据，不对有限基底结果强制归一化。

## 能量和统计误差

沿用训练哈密顿量，能量单位为 $\hbar\omega_c$，保存总能量而非每电子能量。圆盘包含电子与背景圆盘的相互作用，不加背景自能。球面主结果仅包含电子间库仑势。环面沿用包含中性背景和周期自像项的周期库仑求和。

$$
O_{\mathrm{loc}}(R)=\frac{\hat O\Psi(R)}{\Psi(R)},\qquad
\operatorname{SE}(\bar O)=\sqrt{\frac{\sum_{b=1}^B(\bar O_b-\bar O)^2}{B(B-1)}}.
$$

默认每 100 步为一个块，同时计算块长为 50、200、500 时的标准误差和自相关时间。若误差随块长明显变化，汇总标为暂定误差。总能量的误差直接从总能量序列计算。复数物理量分别记录实部和虚部的均值及误差。

## 朗福林参考态

圆盘和球面复用现有解析态：

$$
\Psi_{\mathrm d}\propto\prod_{i<j}(z_i-z_j)^3e^{-\sum_i|z_i|^2/4},\quad z=x-iy,
\qquad\Psi_{\mathrm s}\propto\prod_{i<j}(u_iv_j-u_jv_i)^3.
$$

环面令 $z_i=L_1(u_i+\tau v_i)=x_i+iy_i$，$Z=\sum_i z_i/L_1$：

$$
\Psi_s=e^{\sum_i(-y_i^2/2+ix_iy_i/2)}
\vartheta\begin{bmatrix}1/2+s/3\\1/2\end{bmatrix}(3Z\mid3\tau)
\prod_{i<j}\vartheta\begin{bmatrix}1/2\\1/2\end{bmatrix}
\left((z_i-z_j)/L_1\mid\tau\right)^3,\quad s=0,1,2,
$$

$$
\vartheta\begin{bmatrix}a\\b\end{bmatrix}(w\mid\tau)
=\sum_{k\in\mathbb Z}e^{i\pi\tau(k+a)^2+2\pi i(k+a)(w+b)}.
$$

正式计算默认取 $s=0$。实现支持全部质心态，按现有单电子轨道约定验证两个磁性边界相位。表达式中的质心因子参考[环面朗福林态研究](https://mural.maynoothuniversity.ie/id/eprint/10533/1/JS-Energy-2016.pdf)。六电子最低朗道能级的动能应为 $3\hbar\omega_c$。

精确对角化的三个参考能量由用户提供。导入入口为 `scripts/hall_evaluation.py import-references INPUT.yaml OUTPUT.json`，输入需说明电子数、磁通、相互作用强度、能量单位、背景约定、基底、角动量或动量空间及来源。参数或能量零点不一致时停止导入。

## 球面对关联

球面对关联保留 200 个等宽角度分箱，统计无权重电子对计数 $C_b$。累计 $M$ 个构型后，使用精确球面面积归一化：

$$
p_b=\frac{\cos\theta_b-\cos\theta_{b+1}}{2},\qquad
g_b=\frac{2C_b}{MN^2p_b},\qquad g_b^{\mathrm{finite}}=\frac{N}{N-1}g_b.
$$

两者无关联基准分别为 $(N-1)/N$ 和 1。检查总计数为 $MN(N-1)/2$，且 $\sum_b p_bg_b=(N-1)/N$。

估计器逐步保存每个构型的平均计数。后处理按实际行走者数恢复整数计数，并用 64 位整数汇总。每 100 步分块计算标准误差，同时比较 50、200、500 步的分块结果。`pair_correlation` 输出归一化后的关联函数，`pair_correlation:counts_per_configuration` 是原始逐步平均计数。

独立重新评估的入口为：

```bash
.venv/bin/python scripts/sphere_pair_correlation.py run \
  --data-root /home/jqguo/project/data \
  --output /home/jqguo/project/yuhang/runs/RUN_ID \
  --batch-size 4096 --steps 10000 --burn-in 1000
```

`scripts/evaluate_sphere_pair_correlation.sh` 请求一个节点、两张显卡和八个处理器核心。评估固定最终训练参数，先运行三步联合检查，再开始正式采样。`collect` 子命令等待作业结束、下载并核对文件，随后备份合并数据集并更新球面对关联。逐步计数、两种归一化及误差保存在 `sphere/pair_correlation`，原始异常加权结果保留在 `legacy_weighted` 子组中。

提交入口为 `scripts/submit_sphere_pair_correlation.sh`。它先设置项目、数据和结果目录，再通过 `--export=ALL` 传递完整环境，避免提交时额外获取登录环境。设置 `RUN_ID` 后重复调用同一入口会返回已有作业编号。

## Final four-layer networks

The final workflow archives verified training directories and evaluates the disk
and sphere networks with explicit checkpoint paths. The torus group and compatible
reference energies are reused from the existing merged dataset.

```bash
bash scripts/final_observables.sh archive
bash scripts/final_observables.sh submit
bash scripts/final_observables.sh status
bash scripts/final_observables.sh download
bash scripts/final_observables.sh finish
```

Each evaluation job requests four GPUs and twelve CPU cores on one node, with a
48-hour time limit. Regular evaluations use 4096 walkers, 1000 burn-in steps and
10000 recorded steps. Overlap evaluations use 4096 walkers, 100 burn-in steps and
2000 recorded steps for every saved checkpoint. A short calculation checks the
same network and estimator configuration before formal sampling.

`hall_evaluation.py run` accepts `--train-dir`, `--trained-only` and
`--disk-spatial`. The sphere pair runner and Fubini preparation also accept
`--train-dir`. Existing defaults remain available for historical workflows.

The workflow stores job identifiers immediately and does not submit an already
recorded task again. `monitor` checks job accounting, downloads complete results,
verifies file checksums, builds the dataset and reproduces the selected figures.
A failed task is recorded without automatic resubmission or parameter changes.
Results and provenance are stored under `/home/ubuntu/work/final_data`.

Completed non-overlap data can be published and plotted while checkpoint-distance
jobs are still running:

```bash
bash scripts/final_observables.sh download
bash scripts/final_observables.sh finish --without-fubini
```

This command verifies the four completed non-overlap tasks, imports their new
sphere/disk measurements, and reproduces the 24 figures that do not show state
distances. The new sphere/disk Fubini fields remain empty and are listed as
pending in the validation report. The torus group is copied unchanged. A later
full `finish` rebuilds the dataset with verified overlap results and produces all
28 figure pairs. A publication without overlap data is not reused as a complete
publication.
