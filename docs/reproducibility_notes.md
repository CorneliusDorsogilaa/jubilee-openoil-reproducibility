# Reproducibility notes

## Production versus sensitivity numerics

The production configuration uses `N = 1000` and a 30 min transport timestep. Timestep sensitivity was a separate C3 screening campaign at `N = 2000`, `Kh = 100 m² s⁻¹`, with 5, 10, 15, and 30 min timesteps and three seeds.

Particle-count robustness was evaluated with 20 seeds at `N = 500, 1000, 2000, 4000`. Ensemble-mean A90 values were approximately 732.77, 768.43, 789.07, and 809.85 km², with adjacent mean changes of 4.87%, 2.69%, and 2.63%. The paper therefore treats `N = 1000` as the adopted production resolution under the stated diagnostic criterion, not as a universal minimum converged particle count.

## Kh provenance

The production horizontal diffusivity is `Kh = 100 m² s⁻¹`. Some archived NetCDF metadata queries return `None` for the corresponding attribute. The authoritative value is the executed run configuration plus the file naming / manifest record.

## Mixed-unit temperature handling

The archived temperature array used by the mechanism reconstruction contains Celsius-scale records and a Kelvin-scale final record. The reconstruction converts values elementwise when `T > 100` before calling the seawater-density routine. A whole-array median test is not sufficient and produces a spurious final-record Ohnesorge value.

## Evaporation cutoff

The installed OpenOil NOAA weathering implementation skips further evaporation once all active surface elements are older than 24 h. With a 6 h continuous release, the cumulative evaporated mass therefore plateaus at approximately 30 h simulation time.

## Oil records

The executable oil records are NOAA ADIOS:

- AD01483 — Qua Iboe
- AD01440 — Bonny Light
- AD01442 — Cabinda Blend

These are discrete property contrasts. They do not define a probabilistic confidence interval for Jubilee crude.

## External data

GLORYS12V1 and ERA5 forcing data are not redistributed here. Record provider, product, variables, dates, sampling, and retrieval details in the final archive metadata.


## Software environment freeze

The final repository contains two Conda environment records. `environment.yml` is a compact recreation specification for the principal runtime dependencies. `environment_exact.yml` is the exact exported `jubilee-openoil-g2` environment used for the final analyses, with the machine-specific local Conda `prefix:` removed before publication. The exact export records Python 3.11.16 and OpenDrift 1.14.12.
