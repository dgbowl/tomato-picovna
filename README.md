# tomato-picovna

`tomato` driver for Pico Technologies PicoVNA network analysers.

This driver is a wrapper around the [`vna`](https://pypi.org/project/vna/) library, which is part of the [PicoVNA 5 SDK](https://github.com/picotech/picovna5-examples). As such, the driver needs to be supplied with a location of the SDK using the *settings file* of `tomato`.

## Installation
1. Download and install the PicoVNA 5 software. Tested with the following versions:
  - `5.3.3`
  - `5.3.5`
2. Download the `vna.py` file and save it in the `lib` folder within the PicoVNA 5 installation directory:
  - On Linux, this is `/opt/picovna/lib`.
3. Pass the location of the `lib` folder within the installation directory (e.g. `/opt/picovna/lib`) as `settings['dllpath']` to the driver.

## Supported functions

### Capabilities
- `linear_sweep` for performing a sweep of the reflection coefficient using linearly spaced points

### Attributes
- `temperature`, the temperature of the PicoVNA device, `RO`, `float`
- `bandwidth`, the filter bandwidth for the sweep in Hz, `RW`, `float`
- `power_level`, the power amplitude of the sweep in dBm, `RW`, `float`
- `sweep_params`, a `Sweep` defining the acquisition parameters, containing the following attributes:
  - `start`: low frequency of the sweep, in Hz, `RW`, `Quantity`
  - `stop`: high frequency of the sweep, in Hz, `RW`, `Quantity` 
  - `points`: number of points, `RW`, `int ∈ POINTS_SET`
  - `step`: point step size in Hz, `RW`, `Quantity`
  Note that either `points` or `step` can be supplied, not both.
- `sweep_nports`, the number of ports to be swept, selecting a reflection (`= 1`) or transmission (`= 2`) experiment, `RW`, `int`
- `calibration`, the path to the calibration file to be loaded before acquisition

## Contributors

- Peter Kraus
