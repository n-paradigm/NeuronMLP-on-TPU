
import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,'/home/zouji')
root=Path('/home/zouji/svdflash_tpu/optimization_1_4_20260916/20260916T134014Z')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
import svdflash_tpu
prof=load('svdflash_tpu._optimization_profiler',root/'profiler.py')
svdflash_tpu.profiler=prof;sys.modules['svdflash_tpu.profiler']=prof
candidate=load('_final_candidate',root/'candidate.py')
prof.kernels=candidate
bench=load('svdflash_tpu._optimization_compare',root/'compare_dma.py')
result=bench.main(['--baseline',str(root/'baseline.py'),'--candidate',str(root/'candidate.py'),
 '--model-path','/home/zouji/models/llama3-8b_0.9_svd','--layer','0','--seq-len','128',
 '--warmup','5','--repeats','30','--output',str(root/'real_layer0_S128.json')])
print('REAL_SPEEDUP',result['cases']['checkpoint']['speedup_vs_saved'],flush=True)
import jax,jax.numpy as jnp,numpy as np
from svdflash_tpu.bench.checkpoint_speedup import load_mlp_factors
ug,vg,uu,vu,ud,vd=next(iter(load_mlp_factors('/home/zouji/models/llama3-8b_0.9_svd',[0])))
factors=tuple(a.astype(jnp.bfloat16) for a in (uu,vu,ug,vg,ud,vd))
x=jnp.asarray(np.random.default_rng(1).standard_normal((512,4096),dtype=np.float32),jnp.bfloat16)
print('VERIFY_REAL_S512',flush=True)
fn=jax.jit(candidate.svd_mlp_with_fused_kernel)
with jax.default_matmul_precision('highest'):
 reference=jax.jit(candidate._unfused_reference)
 expected=reference(x,*factors)
got=fn(x,*factors)
verification=bench.errors(got,expected)
assert verification['relative_l2'] < 0.02,verification
print('REAL_S512_CHECK',verification,flush=True)
(root/'real_layer0_S512_check.json').write_text(json.dumps(verification,indent=2))
# Verify the changed profiler sees native MXU scopes and async DMA markers.
rng=np.random.default_rng(3)
def rand(shape):return jnp.asarray(rng.standard_normal(shape,dtype=np.float32)*0.02,jnp.bfloat16)
small=(rand((3,257)),rand((257,513)),rand((513,513)),rand((257,513)),rand((513,513)))
params=prof.tile_params('1,2,1,1')
scoped=prof.instrumented_kernels()
traced,_=prof._compile(lambda *a:scoped.fused_mlp_up_T(*a,**params),small,1,prof.DEVICE_TRACE_OPTIONS)
plain=jax.jit(lambda *a:candidate.fused_mlp_up_T(*a,**params))
np.testing.assert_array_equal(np.asarray(traced(*small)),np.asarray(plain(*small)))
entry=prof.capture('gate-up',traced,small,root/'profiler_smoke',1,True)
inv=entry['inventory']
assert inv['device_event_count'] > 0,inv
assert any('DMA/weights/wait' in k for k in inv['marker_event_counts']),inv
assert any('/MXU/' in k for k in inv['marker_event_counts']),inv
(root/'profiler_smoke.json').write_text(json.dumps(entry,indent=2))
print('PROFILER_CHECK',inv,flush=True)
print('FINAL_VALIDATION_COMPLETE',root,flush=True)
