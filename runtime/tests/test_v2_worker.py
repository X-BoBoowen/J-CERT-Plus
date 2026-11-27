"""Tiny harness checks, never an additional graph performance batch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy import sparse
from src import v2_runner as v2
from src.access_bench import matrix_hashes


class V2Worker(unittest.TestCase):
    def test_real_tiny_service_and_root_profile_respect_hard_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            graph=root/'graph.npz';sparse.save_npz(graph,sparse.csr_matrix((4,4),dtype=np.int64))
            fixture=root/'inputs.npz';matrix=np.array([[65536,-65536,32768,-32768]]*4,np.int64)
            np.savez(fixture,opinions=matrix)
            mh,rh=matrix_hashes(matrix,(4,4))
            spec=dict(n=4,catalog_size=6,weighted_path=str(graph),weighted_sha256=v2.sha(graph))
            record=dict(graph='tiny',stream='independent',file='inputs.npz',npz_sha256=v2.sha(fixture),
                        matrix_sha256=mh,opinion_hashes=rh,shape=[4,4])
            v2.save(root/'PROTOCOL.json',{})
            v2.save(root/'INPUTS.json',dict(protocol_sha256=v2.sha(root/'PROTOCOL.json'),
                 graphs={'tiny':spec},fixtures=[record],code={'source_files_sha256':v2.sources()}))
            for stage in ('A','B1'):
                job=root/f'{stage}.job.json';out=root/f'{stage}.out.json'
                v2.save(job,dict(stage=stage,graph='tiny',stream='independent',method='H',requests=1,
                                budget=15,memory_bytes=2*1024**3,frozen_manifest_sha256=v2.sha(root/'INPUTS.json')))
                code=('from pathlib import Path;from src import v2_runner as v;'
                      f'v.V2=Path({str(root)!r});'
                      "v.validate_job=lambda *args:None;"
                      "v.load_freeze=lambda: ({},__import__('json').loads((v.V2/'INPUTS.json').read_text()));"
                      f'v.worker(Path({str(job)!r}),Path({str(out)!r}))')
                env=os.environ|{k:'1' for k in v2.THREAD_ENV}
                run=subprocess.run([sys.executable,'-c',code],cwd=v2.ROOT,env=env,text=True,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout)
                result=json.loads(out.read_text())
                self.assertIn(result['status'],('certified_stop','completed_budget','profile_completed'),result)
                self.assertEqual(result['hard_limit'],2*1024**3)
                self.assertGreater(result['metering']['solve_calls'],0)
                self.assertEqual(result['affinity'],[0])


if __name__=='__main__':unittest.main()
