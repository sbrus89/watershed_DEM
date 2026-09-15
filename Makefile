# Delaware basin 1/9" topobathy DEM pipeline.
#
# Stages are idempotent; sentinels under manifests/ let make skip completed
# ones. `make -j` parallelism is NOT used across stages (they're sequential);
# compositing parallelizes internally via scripts/08 + GNU parallel.

PY := python
NPROC := 6
HUC4 := $(shell $(PY) -c "import dem_pipeline as dp; print(dp.load_config()['project']['huc4'])")

.PHONY: all env test test-slice boundary audit discover download vrts ehydro \
        watermask composite cogs qa overview clean-sentinels

all: qa overview

env:
	mamba env create -f environment.yml || mamba env update -f environment.yml

test:
	pytest || $(PY) tests/run_tests.py

test-slice:
	bash scripts/run_test_slice.sh

manifests/01_boundary.done:
	$(PY) scripts/01_get_boundary.py && touch $@
boundary: manifests/01_boundary.done

manifests/02_audit.done: manifests/01_boundary.done
	$(PY) scripts/02_audit_cudem_grid.py && touch $@
audit: manifests/02_audit.done

manifests/03_discover.done: manifests/02_audit.done
	$(PY) scripts/03_discover_tiles.py && touch $@
discover: manifests/03_discover.done

manifests/04_download.done: manifests/03_discover.done
	$(PY) scripts/04_download.py && touch $@
download: manifests/04_download.done

manifests/05_vrts.done: manifests/04_download.done
	$(PY) scripts/05_build_source_vrts.py && touch $@
vrts: manifests/05_vrts.done

manifests/06_ehydro.done: manifests/04_download.done
	$(PY) scripts/06_process_ehydro.py && touch $@
ehydro: manifests/06_ehydro.done

manifests/07_watermask.done: manifests/04_download.done manifests/06_ehydro.done
	$(PY) scripts/07_build_water_mask.py && touch $@
watermask: manifests/07_watermask.done

manifests/08_composite.done: manifests/05_vrts.done manifests/06_ehydro.done manifests/07_watermask.done
	jq -r '.features[].properties.tile' config/tiles_$(HUC4).geojson | \
	  parallel -j$(NPROC) $(PY) scripts/08_composite_tile.py --tile {} && touch $@
composite: manifests/08_composite.done

manifests/09_cogs.done: manifests/08_composite.done
	$(PY) scripts/09_make_cogs_vrt.py && touch $@
cogs: manifests/09_cogs.done

qa: manifests/09_cogs.done
	$(PY) scripts/10_qa_checks.py

# Inspection only -- nothing here feeds the product (see plot_overview.py's
# docstring). Depends on the cogs sentinel (not the phony `qa` target) so a
# `make all` rerun with nothing new to compose skips this too; the Esri
# basemap tiles it fetches are cached under
# data/final/quicklooks/.basemap_cache/, so a rerun after that cache is warm
# is fast even though it isn't gated on `qa` having actually run.
manifests/11_overview.done: manifests/09_cogs.done
	$(PY) scripts/plot_overview.py && touch $@
overview: manifests/11_overview.done

clean-sentinels:
	rm -f manifests/*.done
