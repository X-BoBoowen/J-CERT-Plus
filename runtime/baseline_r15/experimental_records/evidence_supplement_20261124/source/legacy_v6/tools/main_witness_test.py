"""Tamper and independent rational-oracle checks for complete decision logs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import sympy as sp
from scipy import sparse
import main_witness_emit
import main_witness
from main_witness_input import DATA, bind_gain_intervals, complete, expected, instance
from main_witness_verify import verify


class MainWitnessTest(unittest.TestCase):
    def test_published_rows_must_be_complete_and_distinct(self):
        keys={("0","full"),("0","adaptive")}
        rows=[dict(query="0",method=m) for m in ("full","adaptive")]
        complete(rows,("query","method"),keys)
        for altered in (rows[:1],rows+rows[:1],rows[:1]*2):
            with self.assertRaises(ValueError):
                complete(altered,("query","method"),keys)

    def test_external_gain_interval_tamper(self):
        base=dict(graph="Alpha", seed="17", proof_unit=str(1<<40), method="full", epoch="1",
                  directional="1",guarded="1")
        result=dict(denominator=1<<80, gains=[(3,4)], objective=(3,4))
        valid_runs=[dict(base,query=str(q),method=m,gain_low="2",gain_high="5")
                    for q in range(10) for m in ("full","fixed","adaptive")]
        for low, high in ((4,5),(2,3)):
            with patch("main_witness_input.records",
                       side_effect=[[dict(base,low=str(low),high=str(high))],valid_runs]):
                with self.assertRaisesRegex(ValueError,"outside published interval"):
                    bind_gain_intervals("mixed",result)
        altered=[dict(row) for row in valid_runs]
        altered[-1]["gain_low"]="4"
        with patch("main_witness_input.records",
                   side_effect=[[dict(base,low="2",high="5")],altered]):
            with self.assertRaisesRegex(ValueError,"outside published interval"):
                bind_gain_intervals("mixed",result)

    def test_persistent_root_sign_corruption(self):
        g,s,p,b = instance("persistent5")
        plan = expected("persistent5",p)
        with np.load(DATA/"main_witness_persistent5.npz") as f:
            original = {k:f[k] for k in f.files}
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/"w.npz"
            for defect in ("missing","wrong operator"):
                parts = original.copy()
                key = f"root_k{plan[4]}"
                if defect=="missing":
                    del parts[key]
                else:
                    parts[key] = np.zeros_like(parts["root"])
                np.savez(log,**parts)
                with self.assertRaises((ValueError,KeyError)):
                    verify(g,s,p,plan,b,log,all_root_signs=True)

    def test_primary_method_binding(self):
        methods = ("adaptive","full","transport")
        base = dict(graph="Alpha",seed="17",proof_unit=str(1<<48),universe="541539",
                    catalog_visits="541539")
        rows = [dict(base,query=str(q),method=m,epoch=str(t),winner_id=str(t),
                     transport_enabled=str(int(m!="full")))
                for q in range(10) for m in methods for t in range(1,16)]
        with patch("main_witness_input.records",return_value=rows):
            self.assertEqual(expected("universe5"),list(range(1,16)))
        for col,value in (("transport_enabled","0"),("catalog_visits","12"),
                          ("winner_id","120"),("method","full")):
            altered = [dict(r) for r in rows]
            altered[-1][col] = value
            with patch("main_witness_input.records",return_value=altered):
                with self.assertRaises(ValueError):
                    expected("universe5")

    def test_registered_and_indexed_row_corruption(self):
        from main_witness_input import records
        for case, target, field in (("persistent6","linview_registered_persistent","sequence"),
                                    ("universe6","promotion_v6_Alpha_rounds","strict_rows")):
            def altered(name):
                rows = records(name)
                if name.startswith(target):
                    row = next(r for r in rows if r.get("b","20")=="20" and
                               r.get("method","adaptive")=="adaptive")
                    row[field] = "[]" if field=="sequence" else "-1"
                return rows
            with self.subTest(case=case), patch("main_witness_input.records",side_effect=altered):
                with self.assertRaises(ValueError):
                    expected(case)

    def test_routed_gain_corruption(self):
        from main_witness_input import bind_routed_gains
        row = dict(b="20",mode="slots6",winner_lower="[1.5]",winner_upper="[2.0]")
        with patch("main_witness_input.records",return_value=[row]):
            with self.assertRaisesRegex(ValueError,"routed planner gain interval"):
                bind_routed_gains([(1,2)])

    def test_primary_alpha_source(self):
        from promotion_inputs5 import clamped_source
        g,s,_,_ = instance("universe5",emitting=True)
        np.testing.assert_array_equal(s*65536,clamped_source(g,"Alpha",17))

    def test_fresh_process_timing_summary(self):
        results = [json.dumps({"plan":[3], "sample":[t,t+1,t+2]}) for t in range(10)]
        with patch.object(main_witness.subprocess,"check_output",side_effect=results) as launch:
            report = main_witness.run("branch")
        self.assertEqual(launch.call_count,10)
        self.assertTrue(all("--child" in call.args[0] for call in launch.call_args_list))
        self.assertEqual(report["median_seconds"],[4.5,5.5,6.5])
        self.assertEqual(report["iqr_seconds"][0],[2.25,6.75])
        self.assertEqual(report["independent_processes"],10)
        for median,(lower,upper) in zip(report["median_seconds"],report["median_ci95_seconds"]):
            self.assertLess(lower,median)
            self.assertGreater(upper,median)

    def test_weighted_rational_oracle(self):
        matrix=np.array([[0,-.5,0,0],[-.5,0,.25,0],[0,.25,0,.5],[0,0,.5,0]])
        graph=sparse.csr_matrix(matrix)
        pool=(np.array([0,0,1]),np.array([1,2,3]),np.array([.5,.125,.25]),np.array([True,False,False]))
        s=np.array([.1,-.8,.5,-.3])
        A=sp.eye(4)+sp.diag(*[sp.Rational(x) for x in abs(matrix).sum(axis=1)])-sp.Matrix(matrix).applyfunc(sp.Rational)
        v=sp.Matrix([sp.Rational(x) for x in s])
        gains=[]
        for i,j,w,flip in zip(*pool):
            C=A.copy()
            w=sp.Rational(w)
            C[i,j]-=w*(2 if flip else 1)
            C[j,i]=C[i,j]
            if not flip:
                C[i,i]+=w
                C[j,j]+=w
            gains.append((v.T*(A.inv()-C.inv())*v)[0])
        winner=max(range(len(gains)),key=gains.__getitem__)
        with tempfile.TemporaryDirectory() as d, patch.object(main_witness_emit,"DATA",Path(d)), patch.object(
                main_witness_emit,"instance",return_value=(graph,s,pool,1)):
            main_witness_emit.produce("fixture")
            res=verify(graph,s,pool,[winner],1,Path(d)/"main_witness_fixture.npz")
        lo,hi=res["objective"]
        self.assertLessEqual(lo,gains[winner]*res["denominator"])
        self.assertGreaterEqual(hi,gains[winner]*res["denominator"])

    def test_tampering(self):
        g,s,p,b=instance("branch")
        plan=expected("branch")
        with np.load(DATA/"main_witness_branch.npz") as f:
            original={k:f[k] for k in f.files}
        claim=json.loads(original["claims"].tobytes())
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/"w.npz"
            for defect in ("wrong-row","source","catalog","root","missing","extra","policy","tail","hot"):
                with self.subTest(defect=defect):
                    parts=original.copy()
                    ids=plan.copy()
                    opinion=s.copy()
                    pool=tuple(a.copy() for a in p)
                    if defect=="wrong-row":
                        ids[:2]=ids[1::-1]
                    elif defect=="source":
                        opinion*=-1
                    elif defect=="catalog":
                        pool[2][plan[0]]*=2
                    elif defect=="root":
                        parts["root"]=-parts["root"]
                    elif defect=="missing":
                        del parts[f"t0k{plan[0]}"]
                    elif defect=="extra":
                        parts["unreferenced"]=parts["root"]
                    else:
                        changed=[dict(x) for x in claim]
                        field={"policy":"delta","tail":"tail","hot":"hot"}[defect]
                        changed[0][field]=[1] if field=="hot" else 1
                        parts["claims"]=np.frombuffer(json.dumps(changed,sort_keys=True,separators=(",",":")).encode(),np.uint8)
                    np.savez(log,**parts)
                    with self.assertRaises((ValueError,KeyError,OverflowError)):
                        verify(g,opinion,pool,ids,b,log)


if __name__=="__main__":
    unittest.main()
