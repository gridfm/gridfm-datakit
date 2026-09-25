# Environment used for the paper GENCO GPU runtime matrix

Original timed jobs: **954945–954948** (`matrix_aligned_ram_{small,large}`,
`matrix_aligned_get_{small,large}`), submitted ~2026-07-12, LSF
`-gpu num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3 -M 128G -n 40`
on CCC 7xx hosts (`cccxc713` small RAM; `cccxc708` the other three).

Those job logs were never flushed, so versions were **not** captured inside
the July jobs. This file is a **live dump of the same venv on the same host
class**, using that LSF template:

- Job **958888**, 2026-09-25 04:21 local, host **cccxc713** (also the original
  RAM-small host)
- Python: `/u/apu/gridfm_model_evaluation/venv/bin/python`
- Raw dump: [`genco_environment_dump.json`](genco_environment_dump.json)

Package pins match the 2026-07-30 venv snapshot in the evaluation-repo notes.
The **host kernel and NVIDIA driver** may have been patched since July 2026.

## Python runtime (executed)

- **Python 3.12.9** (conda-forge, GCC 13.3.0)
- Executable: `/u/apu/gridfm_model_evaluation/venv/bin/python`

## PyTorch / CUDA (timing-critical)

| Package | Version |
|---------|---------|
| torch | **2.8.0+cu128** |
| torch.version.cuda | **12.8** (toolkit torch was built against) |
| torch.backends.cudnn | **9.10.2** (`cudnn.version() == 91002`) |
| triton | **3.4.0** |
| torchvision | 0.23.0 |
| torchaudio | 2.8.0 |
| torch-geometric | **2.7.0** |
| torch-scatter | **2.1.2+pt28cu128** |

Compile mode used in the matrix: `torch.compile(mode="reduce-overhead")`, 20 warmup batches.
Timed inference is FP32 on one exclusive H100.

### Bundled NVIDIA wheels (from the torch cu128 install)

| Package | Version |
|---------|---------|
| nvidia-cuda-runtime-cu12 | 12.8.90 |
| nvidia-cublas-cu12 | 12.8.4.1 |
| nvidia-cudnn-cu12 | **9.10.2.21** |
| nvidia-nccl-cu12 | 2.27.3 |
| nvidia-cufft-cu12 | 11.3.3.83 |
| nvidia-cusolver-cu12 | 11.7.3.90 |
| nvidia-cusparse-cu12 | 12.5.8.93 |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 |
| nvidia-nvjitlink-cu12 | 12.8.93 |

Host driver from `nvidia-smi` on this dump job (not pinned in the venv):
**NVIDIA-SMI 610.57.04**, CUDA UMD **13.3**. That is the node driver; it is
newer than torch's CUDA **12.8** build.

## Model / data libraries (on the inference path)

| Package | Version |
|---------|---------|
| gridfm-graphkit | **0.0.4** |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| numpy | 2.4.2 |

Present in the same venv but **not** on the timed forward path: Lightning 2.6.1,
pandas 2.3.3, scipy 1.17.0, MLflow, Jupyter, juliacall, gridfm-datakit, etc.

## Hardware (dump job = original host class)

- **NVIDIA H100 80GB HBM3**, exclusive (`torch.cuda.get_device_name` / `nvidia-smi`)
- **AMD EPYC 9634** 84-core, 2 sockets (168 CPUs with SMT)
- LSF: `-M 128G`, `-n 40`, `span[hosts=1]`, 7xx `hname` select (`cccxc702`–`716`)
- OS on dump: RHEL 9.8 kernel `5.14.0-687.34.1.el9_8.x86_64`

PowerModels / Julia pins for the paired classical matrix:
[`../outputs_julia/full_matrix/environment_versions.md`](../outputs_julia/full_matrix/environment_versions.md).
