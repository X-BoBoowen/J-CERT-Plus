"""Three same-domain access strategies; numerical work lives in one service."""
import numpy as np

from .a2_catalog import CandidateBatch, MixedFrontier
from .weighted_adapter import Kernel as ExplicitKernel
from .common import UNIT
from .measurement_policy import validate_policy,light_access_memory
from long_promotion import SCREEN
from universe_index import build_incidence


RATIO=(UNIT//SCREEN)**2
COUNTERS=('frontier_rebuilds','raw_heap_pops','graph_rejects','committed_rejects',
          'processed_skips','insertion_emissions','flips_emitted',
          'stream_pairs_examined','stream_positive_rejects','stream_processed_skips',
          'stream_insertions_emitted','stream_flips_emitted',
          'explicit_raw_visits','emitted_for_screen','node_sort_items',
          'flip_sort_items','flip_cursor_pops','total_source_pops',
          'legal_insertion_pops','processed_insertion_skips','processed_flip_skips')


def batch_bytes(batch):
    return sum(a.nbytes for a in (batch.ids,batch.i,batch.j,batch.flip,batch.weights))


class Access:
    def __init__(self,catalog,method,ledger,*,timers=True,measurement_policy='legacy_deep'):
        self.measurement_policy=validate_policy(measurement_policy)
        self.catalog,self.method,self.timers=catalog,method,timers
        self.explicit=self.inc=self.degree=self.index=None
        self.group_bytes=0
        if method=='D':
            if catalog.m>=2**31:
                raise ValueError('this explicit weighted-index implementation requires M<2^31; canonical IDs remain int64')
            with ledger.phase('catalog_build'):
                pool=catalog.materialize()
                self.explicit=ExplicitKernel(catalog.W,pool)
                del pool
            ledger.times.setdefault('structural_weight_grouping',0.)
            with ledger.phase('structural_weight_grouping'):
                self.group_bytes=self.explicit.prepare_weight_groups().nbytes
            with ledger.phase('posting/index_build'):
                self.inc=build_incidence(*self.explicit.pool[:2],catalog.n)
                self.inc.left=np.ascontiguousarray(self.inc.left,dtype=np.int64)
                self.inc.right=np.ascontiguousarray(self.inc.right,dtype=np.int64)
                self.degree=np.diff(self.inc.left)+np.diff(self.inc.right)
                for a in (self.inc.left,self.inc.right,self.inc.ids,self.degree):a.flags.writeable=False
        self.root_frontier=None
        self.max_batch_bytes=0
        self.max_frontier_bytes=0
        self.frontier_memory={}
        self.root_seed_stats={}
        self.seed_counts={}
        self.seed_build_phases={}

    def begin(self,xc,rc,hot_size,ledger):
        self.index=None;self.root_frontier=None
        self.stats=None;self.frontier=None;self.iterator=None;self.scan_counts={}
        self.max_batch_bytes=0;self.max_frontier_bytes=0;self.frontier_memory={}
        self.root_seed_stats={}
        self.seed_build_phases={}
        self.root_caps_created=0
        if self.method=='D':
            self.root_xc=xc.copy();self.root_rc=int(rc)
            with ledger.phase('caps_build'):
                caps=self.explicit.root_caps(xc,rc)
                self.root_caps_created=len(caps)
            with ledger.phase('caps_sort'):
                k=min(hot_size,self.catalog.m)
                if k:
                    cutoff=np.partition(caps,self.catalog.m-k)[-k]
                    above=np.flatnonzero(caps>cutoff)
                    selected=np.r_[above,np.flatnonzero(caps==cutoff)[:k-len(above)]]
                    i,j,_=self.explicit.pool
                    seeds=i[selected].astype(np.int64)*self.catalog.n+j[selected]
                else:seeds=np.empty(0,np.int64)
            with ledger.phase('posting/index_build'):
                self.index=self.explicit.index(xc,rc,self.inc,self.degree,caps=caps,timers=self.timers)
            del caps
            self.root_seed_rule='paid-global-root-caps-top64/canonical-ties'
            self.seed_build_phases=dict(full_root_caps=ledger.times['caps_build'],
                                        top64_select=ledger.times['caps_sort'])
        else:
            ledger.times.setdefault('frontier_prepare',0.)
            with ledger.phase('frontier_prepare'):
                frontier=MixedFrontier(self.catalog,xc,rc)
            if self.measurement_policy=='legacy_deep':self.observe_frontier(frontier)
            self.seed_build_phases=dict(frontier.build_phases)
            with ledger.phase('seed_access'):
                if hot_size:
                    batch=frontier.pop_batch(hot_size)
                    seeds=batch.ids.copy()
                    self.max_batch_bytes=max(self.max_batch_bytes,batch_bytes(batch))
                else:
                    seeds=np.empty(0,np.int64)
            if self.measurement_policy=='legacy_deep':self.observe_frontier(frontier)
            if self.method=='H':self.root_frontier=frontier
            else:self.root_seed_stats=dict(frontier.counters)
            self.root_seed_rule='paid-joint-frontier-first64/heap-order-recorded'
        seed_batch=self.catalog.batch(seeds)
        self.seed_counts=dict(seed_candidates=len(seeds),seed_screen_rows=len(seeds),
            seed_insertions=int((~seed_batch.flip).sum()),seed_flips=int(seed_batch.flip.sum()))
        return np.asarray(seeds,dtype=np.int64)

    def observe_frontier(self,frontier):
        mem=frontier.memory()
        self.frontier_memory=mem
        value=int(mem.get('total_owned_bytes',0))
        self.max_frontier_bytes=max(self.max_frontier_bytes,value)

    def open_version(self,xc,rc,chosen,excluded,epoch,parent,ledger):
        self.current_xc,self.current_rc=xc,int(rc)
        self.done=False;self.iterator=None;self.frontier=None
        self.scan_counts={}
        self.stats=dict.fromkeys(COUNTERS,0)
        self.stats.update(epoch=epoch,parent_version_id=parent,source=self.method,
                          remaining_upper_coarse=None,source_complete=False)
        self.stats.update({key:0 for key in self.seed_counts})
        if epoch==1:
            self.stats.update(self.seed_counts)
            self.stats['seed_build_phases']=dict(self.seed_build_phases)
            if self.method=='D':
                self.stats['insertion_emissions']=self.seed_counts['seed_insertions']
                self.stats['flips_emitted']=self.seed_counts['seed_flips']
        if self.method=='H':
            if epoch==1:
                self.frontier=self.root_frontier;self.root_frontier=None
            else:
                with ledger.phase('frontier_prepare'):
                    self.frontier=MixedFrontier(self.catalog,xc,rc,chosen=chosen)
            self.stats['frontier_rebuilds']=1
            if self.measurement_policy=='legacy_deep':self.observe_frontier(self.frontier)
        elif self.method=='S':
            self.iterator=self.catalog.iter_blocks(block_size=65536,excluded=excluded,counters=self.scan_counts)
            if epoch==1:
                self.stats.update(self.root_seed_stats)
                self.stats['frontier_rebuilds']=1
        self.ledger=ledger

    def _update(self):
        if self.frontier is not None:
            self.stats.update(self.frontier.counters)
            self.stats['frontier_rebuilds']=1
            self.stats['remaining_upper_coarse']=self.frontier.upper
        self.stats.update(self.scan_counts)

    def next_batch(self,best,excluded):
        if self.done:
            self.stats['source_complete']=True
            return None
        if self.method=='D':
            target=-(-int(best)//RATIO)
            drift=np.abs(self.current_xc-self.root_xc)+self.current_rc+self.root_rc
            local=set(map(int,self.catalog.dense_positions(np.fromiter(excluded,np.int64))))
            ids,choices,costs,tau=self.index.query(drift,target,local)
            i,j,f=self.explicit.pool
            ci,cj=i[ids],j[ids]
            canonical=ci.astype(np.int64)*self.catalog.n+cj
            batch=CandidateBatch(canonical,ci,cj,f[ids],self.explicit.d[ids])
            self.stats.update(explicit_raw_visits=int(costs.min(initial=2**62)),
                remaining_upper_coarse=target-1,tau=int(tau))
            self.stats['insertion_emissions']+=int((~batch.flip).sum())
            self.stats['flips_emitted']+=int(batch.flip.sum())
            self.done=True
        elif self.method=='S':
            batch=next(self.iterator,None)
            self._update()
            if batch is None:
                self.done=True;self.stats['source_complete']=True
                return None
            self.stats['insertion_emissions']+=int((~batch.flip).sum())
            self.stats['flips_emitted']+=int(batch.flip.sum())
        else:
            upper=self.frontier.upper
            target=-(-int(best)//RATIO)
            if upper is None or (best==0 and upper<=0) or upper<target:
                self.done=True;self.stats['source_complete']=True;self._update()
                return None
            batch=self.frontier.pop_batch(256,threshold=target,excluded=excluded)
            self._update()
            if not len(batch):
                self.done=True;self.stats['source_complete']=True
                return None
        self.stats['emitted_for_screen']+=len(batch)
        self.max_batch_bytes=max(self.max_batch_bytes,batch_bytes(batch))
        return batch

    def finish_version(self):
        self._update()
        self.stats['cold_emitted_for_screen']=self.stats['emitted_for_screen']
        if self.frontier is not None:
            if self.measurement_policy=='legacy_deep':self.observe_frontier(self.frontier)
            self.stats['frontier_build_phases']=getattr(self.frontier,'build_phases',{})
        return dict(self.stats)

    def memory(self):
        if self.measurement_policy=='light_v1':return light_access_memory(self)
        explicit_bytes=0
        incidence_bytes=0
        index_private_bytes=0
        if self.explicit is not None:
            explicit_bytes=sum(a.nbytes for a in self.explicit.pool)+self.explicit.d.nbytes
            incidence_bytes=sum(a.nbytes for a in (self.inc.left,self.inc.right,self.inc.ids,self.degree))
        if self.index is not None:
            index_private_bytes=sum(keys.nbytes+ids.nbytes for _,keys,ids in self.index.groups)
        return dict(catalog_materialized=self.method=='D',
            candidate_array_rows=self.catalog.m if self.method=='D' else 0,
            full_root_caps_created_rows=self.root_caps_created,
            m_sized_structures=(['explicit_endpoints/kind','effective_weights','weight_members',
                'incidence_ids','sorted_root_keys/ids','temporary_root_caps'] if self.method=='D' else []),
            explicit_catalog_bytes=explicit_bytes,structural_weight_grouping=self.group_bytes,
            incidence_bytes=incidence_bytes,index_private_bytes=index_private_bytes,
            peak_frontier_owned_bytes=self.max_frontier_bytes,
            peak_emitted_batch_array_bytes=self.max_batch_bytes,
            frontier_snapshot=self.frontier_memory,
            scope='Owned persistent access arrays plus maximum observed frontier/batch payload; '
                  'not total process peak; temporary sorting/dedup/strict/solver work remains in RSS')

    def close(self):
        self.index=None;self.root_frontier=None;self.frontier=None;self.iterator=None
        self.stats=None;self.scan_counts={}
        self.current_xc=None;self.ledger=None
        if hasattr(self,'root_xc'):self.root_xc=None
