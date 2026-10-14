from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton
from triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
  return m;
}
"""


@pytest.mark.parametrize("language", ["c", "c++"])
def test_compile_module(fresh_triton_cache, language):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module", language=language)

    with pytest.raises(TypeError, match="exactly 1 argument"):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module", language=language)
    assert mod2.__file__ == mod.__file__


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_build_python_linker_flags(platform, monkeypatch, tmp_path):
    from triton.runtime import build

    commands = []
    monkeypatch.setattr(build, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(build, "_find_compiler", lambda _: "cc")
    monkeypatch.setattr(build.subprocess, "check_call", lambda cmd, **kwargs: commands.append(cmd))
    build._build("module", "module.c", str(tmp_path), [], [], [], [])
    cmd = commands[0]
    assert ("-bundle" in cmd) == (platform == "darwin")
    assert ("-shared" in cmd) == (platform != "darwin")
    assert ("dynamic_lookup" in cmd) == (platform == "darwin")


def test_wheel_build_reuses_objects_without_leaking_abi_settings(tmp_path):
    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/wheels.yml").read_text()
    script = textwrap.dedent(re.search(r"- name: Build wheels\n.*?run: \|\n(.*?)(?=\n      -)", workflow, re.DOTALL)[1])
    python = tmp_path / "python3"
    python.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1:3] == ['-m', 'cibuildwheel']:
    with open(os.environ['BUILD_LOG'], 'a') as output:
        output.write(json.dumps(dict(os.environ)) + '\\n')
elif len(sys.argv) > 1 and sys.argv[1] == '-c':
    print('pytest')
""")
    python.chmod(0o755)
    (tmp_path / "python").symlink_to(python)
    log = tmp_path / "builds.jsonl"
    subprocess.run(
        ["bash", "-e", "-c", script],
        check=True,
        capture_output=True,
        env=dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            RUNNER_TEMP=str(tmp_path),
            BUILD_ARCH="aarch64",
            BUILD_LOG=str(log),
        ),
    )
    builds = [json.loads(line) for line in log.read_text().splitlines()]
    environments = [dict(token.split("=", 1) for token in shlex.split(build["CIBW_ENVIRONMENT"])) for build in builds]
    assert [env["TRITON_STABLE_ABI"] for env in environments] == ["OFF", "ON", "OFF"]
    assert [env["TRITON_BUILD_PROTON"] for env in environments] == ["ON", "ON", "OFF"]
    for key in ["TRITON_BUILD_DIR", "TRITON_HOME", "CCACHE_DIR"]:
        assert len({env[key] for env in environments}) == 1
        assert environments[0][key].startswith(f"/host{tmp_path}/")
    assert all(env["TRITON_APPEND_CMAKE_ARGS"] == "-DTRITON_BUILD_UT=OFF" for env in environments)
    assert all("ccache" in build["CIBW_BEFORE_ALL"] for build in builds)


def test_compile_module_bad_cache(fresh_triton_cache, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        monkeypatch.setattr(triton.knobs.cache, "manager_class", InvalidFileCacheManager)

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(TypeError, match="exactly 1 argument"):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


@pytest.mark.parametrize("artifact", ["bootstrap", "codegen"])
@pytest.mark.parametrize(
    "mirror", ["", "https://mirror.example/generic/oaiartifacts", "https://mirror.example/generic/oaiartifacts/"])
def test_amd_codegen_download_mirror(artifact, mirror, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setenv("OAIARTIFACTS_BASE_URL", mirror)
    monkeypatch.delenv("TRITON_CACHE_DEPENDENCY_DOWNLOADS", raising=False)
    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "ubuntu-arm64",
            "--llvm-syspath",
            str(tmp_path / "unrelated-core-llvm"),
        ]))

    revision = "123456789abcdef"
    name = f"{'llvm' if artifact == 'bootstrap' else 'amd-codegen'}-{revision[:8]}-ubuntu-arm64-1"
    relative_path = "lib/cmake/llvm/LLVMConfig.cmake" if artifact == "bootstrap" else (
        f"lib/{build_helpers.amd_codegen_library_name()}")
    contents = b"test artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}/{relative_path}")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    archive_bytes = buffer.getvalue()
    checksum = hashlib.sha256(archive_bytes).hexdigest()
    llvm_info = {"llvm_hash": revision, "build_number": 1, "sha256sum": {"ubuntu-arm64": checksum}}
    if artifact == "bootstrap":
        llvm_info["bootstrap_llvm"] = {"build_number": 1, "sha256sum": llvm_info.pop("sha256sum")}

    downloads = []

    def download(url, path, label):
        downloads.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(archive_bytes)

    monkeypatch.setattr(build_helpers, "_download_file", download)
    if artifact == "bootstrap":
        result = Path(build_helpers.download_codegen_llvm("amd", llvm_info, args)) / relative_path
    else:
        (tmp_path / "cmake").mkdir()
        (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
        build_helpers.download_and_copy_amd_codegen(args)
        result = tmp_path / "third_party" / "amd" / "backend" / relative_path

    base = "https://mirror.example/generic/triton-llvm" if mirror else (
        "https://oaitriton.blob.core.windows.net/public/llvm-builds")
    assert downloads == [f"{base}/{name}.tar.gz"]
    assert result.read_bytes() == contents
