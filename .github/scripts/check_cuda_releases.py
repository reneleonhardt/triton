"""Warn about newer official CUDA redistributions without downloading binaries.

>>> newer_releases([(12, 9, 1), (13, 4, 2)], [(12, 9, 2), (13, 4, 2)])
[(12, 9, 2)]
>>> newer_releases([(12, 9, 2)], [(12, 9, 2), (13, 4, 2), (13, 5, 0)])
[(13, 4, 2), (13, 5, 0)]
>>> newer_releases([(13, 4, 2)], [(13, 4, 1), (13, 4, 2)])
[]
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def version(value):
    return tuple(map(int, value.split(".")))


def parse_releases(html):
    """Read stable manifests from NVIDIA's single- or double-quoted links.

    >>> parse_releases("<a href='redistrib_13.4.2.json'>new</a>")
    [(13, 4, 2)]
    >>> parse_releases('<a href="redistrib_12.9.2.json">old</a><a href="redistrib_14.0.0rc1.json">preview</a>')
    [(12, 9, 2)]
    """
    return sorted({version(v) for v in re.findall(r'''href=['"]redistrib_(\d+\.\d+\.\d+)\.json['"]''', html)})


def newer_releases(configured, available):
    newest = {release[:2]: release for release in sorted(available)}
    updates = []
    for minor, release in newest.items():
        pins = [pin for pin in configured if pin[0] == minor[0]]
        if pins:
            changed = minor > max(pin[:2] for pin in pins) or any(pin[:2] == minor and pin < release for pin in pins)
        else:
            changed = minor[0] > max(pin[0] for pin in configured)
        if changed:
            updates.append(release)
    return sorted(updates)


def torch_wheel_channels(release, index, fetch_index, minimum=0):
    """Only report channels with wheels for the exact stable PyTorch release.

    >>> pages = {132: 'torch-2.14.1%2Bcu132-cp314-cp314t-linux_x86_64.whl',
    ...          134: 'torch-2.15.0%2Bcu134-cp314-cp314-linux_x86_64.whl',
    ...          136: 'torch-2.16.0rc1%2Bcu136-cp314-cp314-linux_x86_64.whl'}
    >>> torch_wheel_channels('2.15.0', '<a href="cu132/">old</a><a href="cu134/">new</a><a href="cu136/">preview</a>', pages.__getitem__)
    [134]
    >>> torch_wheel_channels('2.16.0', '<a href="cu136/">preview</a>', pages.__getitem__)
    []
    """
    channels = sorted({int(v) for v in re.findall(r"cu(\d+)/", index)})
    return [
        channel for channel in channels
        if channel > minimum and f"torch-{release}+cu{channel}-" in unquote(fetch_index(channel))
    ]


def warn(message):
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::warning::{message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toolkits", type=Path)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--torch-channels", type=Path)
    parser.add_argument("--torch-version", type=Path)
    args = parser.parse_args()
    try:
        available = parse_releases(fetch(REDIST))
        if not available:
            raise ValueError("NVIDIA's redistribution index contained no stable versions")
        latest = ".".join(map(str, available[-1]))
        print(f"Latest stable CUDA toolkit: {latest}")
        if args.toolkits:
            configured = [
                version(v) for line in args.toolkits.read_text().splitlines() if "cuda-version:" in line
                for v in re.findall(r"\d+\.\d+\.\d+", line)
            ]
            if not configured:
                raise ValueError(f"No CUDA toolkit versions found in {args.toolkits}")
            for release in newer_releases(configured, available):
                warn(
                    f"{args.toolkits}: CUDA {'.'.join(map(str, release))} is newer than the tested SDK matrix; review Torch and GPU compatibility before updating."
                )
        if args.components:
            manifests = {}
            names = {"ptxas": "cuda_nvcc", "cudacrt": "cuda_crt", "cudart": "cuda_cudart"}
            for name, pinned in json.loads(args.components.read_text()).items():
                # Keep the legacy CUPTI major used for pre-Blackwell GPUs.
                sdk = ".".join(map(str, max(v for v in available if v[0] == version(pinned)[0])))
                if sdk not in manifests:
                    manifests[sdk] = json.loads(fetch(f"{REDIST}redistrib_{sdk}.json"))
                manifest = manifests[sdk]
                component = names.get(name, "cuda_" + name.split("-")[0])
                released = manifest.get(component, {}).get("version")
                if released and version(released) > version(pinned):
                    warn(
                        f"{args.components}: {name} {pinned} has newer stable version {released} in CUDA {sdk}; review architecture compatibility before updating."
                    )
        if args.torch_channels or args.torch_version:
            torch = json.loads(fetch("https://pypi.org/pypi/torch/json"))["info"]["version"]
            print(f"Latest stable PyTorch: {torch}")
            if args.torch_channels:
                text = args.torch_channels.read_text()
                match = re.search(r"torch-cuda:\s*\[([^\]]+)\]", text)
                if not match:
                    raise ValueError(f"No Torch CUDA channels found in {args.torch_channels}")
                configured = list(map(int, re.findall(r"\d+", match[1])))
                channels = torch_wheel_channels(
                    torch, fetch("https://download.pytorch.org/whl/"),
                    lambda channel: fetch(f"https://download.pytorch.org/whl/cu{channel}/torch/"), max(configured))
                for channel in channels:
                    warn(
                        f"{args.torch_channels}: stable Torch {torch} wheels now support cu{channel}; review the dependency wheel matrix."
                    )
                if not channels:
                    print(f"No newer stable PyTorch CUDA wheel channel beyond cu{max(configured)}.")
            if args.torch_version:
                configured = re.findall(r"pytorch_release_(\d+\.\d+\.\d+)", args.torch_version.read_text())
                if not configured:
                    raise ValueError(f"No stable Torch versions found in {args.torch_version}")
                if any(version(pin) < version(torch) for pin in configured):
                    warn(
                        f"{args.torch_version}: stable Torch {torch} is newer than a CI image; review matching ROCm/CUDA images and dependency compatibility."
                    )
            print("Torch wheel channels are independent of the CUDA toolkit patch version.")
    except (OSError, ValueError) as error:
        warn(f"CUDA release availability could not be verified: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
