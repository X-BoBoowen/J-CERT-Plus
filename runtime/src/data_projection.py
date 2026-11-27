"""Frozen projection of original SNAP Bitcoin directed ratings."""
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DIGEST = {"Alpha": "3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76",
          "OTC": "6424ac981dad3a019f697fc1b9fcd85c19d8d9f039797758e9ffb6fea100c373"}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(path):
    latest = {}
    nodes = set()
    loops = ties = raw_rows = 0
    cutoff = None
    with gzip.open(path, "rt", newline="") as f:
        for raw_rows, (source,target,rating,stamp) in enumerate(csv.reader(f),1):
            i,j,r = int(source),int(target),int(rating)
            t = Decimal(stamp)
            if not -10 <= r <= 10 or not t.is_finite():
                raise ValueError(f"invalid rating or timestamp on row {raw_rows}")
            cutoff = max(t,cutoff) if cutoff is not None else t
            nodes.update((i,j))
            if i == j:
                loops += 1
                continue
            key = (i,j)
            if key in latest:
                old,_ = latest[key]
                if t < old:
                    continue
                ties += t == old
            latest[key] = t,r
    nodes = sorted(nodes)
    ids = {node:k for k,node in enumerate(nodes)}
    ratings = {}
    for (i,j),(_,r) in latest.items():
        ratings.setdefault(tuple(sorted((i,j))),[]).append(r)
    row, col, values = [], [], []
    zero = mixed = reciprocal = 0
    for (i,j), rr in sorted(ratings.items()):
        reciprocal += len(rr)==2
        mixed += min(rr)<0<max(rr)
        w = sum(rr)/len(rr)
        if w == 0:
            zero += 1
            continue
        a,b = ids[i],ids[j]
        row.extend((a,b))
        col.extend((b,a))
        values.extend((w,w))
    weighted = sparse.csr_matrix((np.array(values,dtype=np.float64),(row,col)),shape=(len(nodes),len(nodes)))
    weighted.sort_indices()
    unit = weighted.copy()
    unit.data = np.sign(unit.data)
    info = dict(rows=raw_rows,cutoff_timestamp=str(cutoff),vertices=len(nodes),self_loop_rows=loops,
                superseded_rows=raw_rows-loops-len(latest),tie_break_rows=ties,
                retained_directions=len(latest),reciprocal_pairs=reciprocal,opposed_pairs=mixed,
                dropped_net_zero_pairs=zero,edges=weighted.nnz//2,
                negative_edges=int((weighted.data<0).sum()//2),
                isolated_vertices=int(np.sum(np.diff(weighted.indptr)==0)),
                magnitude_histogram={str(v):int((abs(weighted.data)==v).sum()//2)
                                     for v in np.unique(abs(weighted.data))})
    return nodes,unit,weighted,info


def main():
    for name,digest in DIGEST.items():
        filename = f"soc-sign-bitcoin{name.lower()}.csv.gz"
        path = ROOT/"data/raw"/filename
        if checksum(path) != digest:
            raise ValueError(f"{filename} source checksum")
        nodes, unit, weighted, info = project(path)
        dest = ROOT/"data/projected"
        dest.mkdir(parents=True,exist_ok=True)
        nodefile = dest/f"{name}_nodes.csv"
        with nodefile.open("w",newline="") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerow(("projected_vertex","original_vertex"))
            writer.writerows(enumerate(nodes))
        info.update(raw_sha256=digest,raw_bytes=path.stat().st_size,node_map_sha256=checksum(nodefile),
                    source_url=f"https://snap.stanford.edu/data/{filename}",
                    description_url=f"https://snap.stanford.edu/data/soc-sign-bitcoin-{name.lower()}.html",
                    projection="entire released snapshot; last timestamp per direction; final file row on timestamp ties; drop self-loops; average observed directions, drop exact zero; preserve node universe sorted by numeric ID",
                    alpha=1,insertion_weight="1/16",flip_weight="absolute projected edge weight")
        for label,graph in (("weighted",weighted),("unit",unit)):
            target = dest/f"{name}_{label}.npz"
            sparse.save_npz(target,graph)
            info[f"{label}_sha256"] = checksum(target)
        (dest/f"{name}_metadata.json").write_text(json.dumps(info,sort_keys=True,indent=2)+"\n")
        print(name,info["vertices"],info["edges"],info["negative_edges"],info["weighted_sha256"])


if __name__ == "__main__":
    main()
