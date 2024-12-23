# tomato-picovna

`tomato` driver for Pico Technologies PicoVNA network analysers.

This driver is a wrapper around the [`vna`](https://pypi.org/project/vna/) library, which is part of the [PicoVNA 5 SDK](https://github.com/picotech/picovna5-examples). As such, the driver needs to be supplied with a location of the SDK using the *settings file* of `tomato`.

## Supported functions

### Capabilities
- `linear_sweep` for performing a sweep of the reflection coefficient using linearly spaced points

### Attributes
- `temperature`, the temperature of the PicoVNA device, `RO`, `float`
- `bandwidth`, the filter bandwidth for the sweep in Hz, `RW`, `float`
- `power_level`, the power amplitude of the sweep in dBm, `RW`, `float`
- `sweep_params`, a list of sweep parameters defining the frequecies of the `start`, `stop`, and `step` for each component of the sweep, in Hz, `RW`, `float`
- `sweep_nports`, the number of ports to be swept, selecting a reflection (`= 1`) or transmission (`= 2`) experiment, `RW`, `int`

## Contributors

- Peter Kraus
