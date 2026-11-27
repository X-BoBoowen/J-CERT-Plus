#!/usr/bin/env python3
"""Overlay verified packages into a fresh destination; refuse changed files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("destination", type=Path)
p.add_argument("archives", nargs="+", type=Path)
a = p.parse_args()
a.destination.mkdir(parents=True, exist_ok=True)
if a.destination.is_symlink():
    raise SystemExit("Destination may not be a symlink")
for archive in a.archives:
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            raise SystemExit("Duplicate ZIP members")
        manifests = [name for name in names if name in {
            "MANIFEST_CODE.json", "MANIFEST_EVIDENCE.json", "MANIFEST_INPUTS.json"}]
        if len(manifests) != 1:
            raise SystemExit("Expected exactly one internal manifest")
        manifest = json.loads(z.read(manifests[0]))
        expected = {e["path"]: e for e in manifest["members"]}
        if set(names) != set(expected) | {manifests[0]}:
            raise SystemExit("Member set differs from manifest")
        bad = z.testzip()
        if bad is not None:
            raise SystemExit("CRC failure: " + bad)
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or "\\" in name
                    or ":" in path.parts[0] or info.is_dir()
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                raise SystemExit("Unsafe ZIP path: " + name)
            out = a.destination.joinpath(*path.parts)
            for parent in out.parents:
                if parent == a.destination.parent:
                    break
                if parent.is_symlink():
                    raise SystemExit("Symlink in destination: " + str(parent))
            raw = z.read(name)
            actual = digest_bytes(raw)
            if name in expected and (len(raw) != expected[name]["bytes"]
                                      or actual != expected[name]["sha256"]):
                raise SystemExit("Payload hash mismatch: " + name)
            if out.exists():
                if not out.is_file() or out.stat().st_size != len(raw) or digest_file(out) != actual:
                    raise SystemExit("Refusing nonidentical overwrite: " + str(out))
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as f:
                f.write(raw)
print("Verified package overlay completed")
