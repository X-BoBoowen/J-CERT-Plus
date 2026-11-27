"""Explicit reporting policies; no score, cache or resource-admission decisions."""

POLICIES=('legacy_deep','light_v1')


def validate_policy(value):
    if value not in POLICIES:
        raise ValueError('unknown measurement policy: '+str(value))
    return value


def worker_policy(frozen,job):
    if 'measurement_policy' not in job:
        raise ValueError('worker policy must be explicit in every frozen job')
    policy=validate_policy(job['measurement_policy'])
    kind=frozen.get('run_kind')
    if kind=='formal':
        if frozen.get('measurement_policy')!='light_v1' or policy!='light_v1':
            raise ValueError('formal V4 workers require matching light_v1 policy')
    elif kind=='invariance':
        if frozen.get('measurement_policy')!='explicit_per_job':
            raise ValueError('diagnostic policy must be declared per job')
    else:
        raise ValueError('unknown V4 worker kind')
    return policy


def payload(array):
    return None if array is None else int(array.nbytes)


def access_snapshot(access):
    """A fixed set of len/nbytes reads; never traverses container members."""
    frontier=getattr(access,'frontier',None)
    if frontier is None:frontier=getattr(access,'root_frontier',None)
    blocks=getattr(access,'blocks',None)
    return dict(
        h_heap_items=None if frontier is None else len(frontier.heap),
        h_flip_items=None if frontier is None else len(frontier.flips),
        h_order_array_payload=None if frontier is None else payload(frontier.order),
        h_center_array_payload=None if frontier is None else payload(frontier.xc),
        g_root_flip_items=None if not hasattr(access,'root_flips') else len(access.root_flips),
        g_flip_order_items=None if not hasattr(access,'flip_order') else len(access.flip_order),
        g_block_heap_items=None if blocks is None else len(blocks.heap),
        g_node_summary_items=None if blocks is None else len(blocks.nodes),
        g_order_array_payload=None if blocks is None else payload(blocks.order),
        g_root_diagonal_payload=payload(getattr(access,'root_diagonal',None)),
        g_current_diagonal_payload=payload(getattr(access,'diagonal',None)),
        current_center_payload=payload(getattr(access,'current_xc',None)),
        peak_emitted_batch_array_bytes=int(access.max_batch_bytes))


def light_access_memory(access):
    return dict(catalog_materialized=access.method=='D',
        candidate_array_rows=access.catalog.m if access.method=='D' else 0,
        full_root_caps_created_rows=access.root_caps_created,
        m_sized_structures=[],explicit_catalog_bytes=None,structural_weight_grouping=None,
        incidence_bytes=None,index_private_bytes=None,peak_frontier_owned_bytes=None,
        peak_emitted_batch_array_bytes=access.max_batch_bytes,frontier_snapshot=None,
        memory_observation_schema='light_v1',
        scope='Fixed-count lengths, named array payloads and existing counters only; '
              'deep object bytes/watermarks unobserved; shared payloads must not be summed as owned process memory')


class LightObservations:
    def __init__(self):
        self.rows=[]

    def observe(self,point,epoch,access,views,fac,hot,seen,chosen,decisions,access_rows):
        # The number of fields is independent of graph, cache, heap and history
        # sizes. Appending a fixed-size record does not inspect past records.
        row=dict(point=point,epoch=int(epoch),gram_items=len(views.entries),
                 low_rank_step_count=len(fac.steps),hot_items=len(hot),seen_items=len(seen),
                 chosen_items=len(chosen),decision_items=len(decisions),access_event_items=len(access_rows))
        row.update(access_snapshot(access))
        self.rows.append(row)
