# Local evaluation runs

Raw recordings, generated media, frozen snapshots and historical experiments
stay here for reproduction. They are excluded from package installation and Git
by default. Do not use this directory as an application source library.

The maintained implementation is `../dartboard_vision/`. In the real-sample
directory, old module names for masks, contacts, session state, tracking,
alignment and fusion are compatibility imports of that package. Existing local
drivers remain callable after installing the package with `pip install -e .`.

The portable replay command and recording-specific configuration are documented
in `../examples/recorded_rig/README.md`. Completed recordings and results are
retained; cleanup does not delete the original evidence.
