#!/bin/bash
# Region-02 channel burn on a Perlmutter CPU compute node (shared QOS).
# Resumable: network manifests and finished tiles are skipped on resubmit.
#   sbatch scripts/burn_region.sbatch            # from the repo dir
#SBATCH --job-name=burn_r02
#SBATCH --account=m4457
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=logs/burn_r02_%j.log

set -euo pipefail
cd /pscratch/sd/s/sbrus/DE_watershed_DEM
ENV=/global/cfs/cdirs/e3sm/sbrus/miniforge3/envs/de-dem
export PATH=$ENV/bin:$PATH
export PROJ_DATA=$($ENV/bin/python -c "import pyproj;print(pyproj.datadir.get_data_dir())")

echo "start $(date) on $(hostname)"
python scripts/burn_region.py all --nproc 8
python scripts/burn_region.py qa --connectivity
echo "done $(date)"
