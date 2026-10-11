# Dependency compatibility

CI builds a GIL CPython 3.11 wheel and a CPython 3.12 abi3 wheel tested through
3.15. Experimental free-threaded compiler wheels use cp314t/cp315t and
nanobind's free-threaded runtime; they cannot use `TRITON_STABLE_ABI=ON`.

Free-threaded builds require `TRITON_BUILD_PROTON=OFF`. CPU tests check GIL-free
imports and concurrent specialization, dialect loading and LLVM bindings.
Each thread must own its mutable MLIR contexts/modules. CUDA/HIP launchers
still require the GIL; Proton and GPU free-threaded execution remain unvalidated.

CUDA 13 components match SDK 13.4.2; pre-Blackwell CUPTI uses CUDA 12.9.
Native GPU CI validates execution/profiling. The advisory checks stable SDK
components and Torch releases without automatically changing them.

Nanobind requires >=3.1.0, with matching library/binding headers. CMake 4.4.4
builds pass. LLVM stays on the required SDK revision: LLVM 23.1.3 lacks this
checkout's MLIR resource parameters, so updating needs a compatible SDK build.
Stable ROCm images keep their bundled Torch; development images cover new hardware.

Workflow self-references require Actions runner >=2.336.0 on self-hosted runners.
