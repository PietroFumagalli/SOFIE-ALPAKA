# SOFIE-ALPAKA

Kernels for heterogeneous architectures written in [Alpaka](https://alpaka.readthedocs.io/en/stable/) (An Abstraction Library for Parallel Kernel Acceleration) for [SOFIE](https://github.com/ML4EP/SOFIE) (System for Optimised Fast Inference code Emit).

This repository does not depend on SOFIE, but these kernels will eventually go into SOFIE.

Submission for CS-433: Machine Learning; hopefully, will not stay just as a random course project, but will become a part of the actual ML code written at CERN.

## Dependencies

- `Alpaka` (`1.2.0`): for heterogenous kernels; present as a git submodule in `external/`
- `Boost` (`libboost-all-dev` on Debian): for Alpaka
- `cmake` and `make`: for building and testing the project

## Usage

Clone the repository with all the submodules (dependencies):

```
git clone https://github.com/Saransh-cpp/SOFIE-ALPAKA --recursive
```

### Building kernels and tests

To build all kernels in `bin/`:

```
make all ALPAKA_ACCELERATOR_FLAG=enable_an_alpaka_accelerator
```

where `ALPAKA_ACCELERATOR_FLAG` defaults to `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED` (enables the CPU threaded backend).

### Running tests

To run all kernel tests (and build if not built before):

```
make test ALPAKA_ACCELERATOR_FLAG=enable_an_alpaka_accelerator
```

where `ALPAKA_ACCELERATOR_FLAG` defaults to `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED` (enables the CPU threaded backend).

### Running integration tests

To run SOFIE integration tests:

```
cd tests/sofie_integration
cmake -S. -Bbuild
cmake --build build
```
