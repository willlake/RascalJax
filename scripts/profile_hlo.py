import os, sys, re, collections
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profile_kernel import I, data, st, key
st_smu = dataclasses.replace(st, n_ell=0)
idx = jnp.arange(5000)
for label, s in (("legendre", st), ("s_mu", st_smu)):
    acc = I.zero_accumulators(s)
    comp = jax.jit(I.accumulate_batch, static_argnames=("st",)).lower(key, idx, data, s, acc).compile()
    txt = comp.as_text()
    ops = collections.Counter(re.findall(r"= \S+ (\S+?)\(", txt))
    print(label, "HLO ops:", {k: v for k, v in ops.most_common(12)})
    big = re.findall(r"f64\[(\d+)(?:,(\d+))?(?:,(\d+))?\]", txt)
    print(label, "memory stats:", comp.memory_analysis())
    open(f"/tmp/claude-102016/-global-u2-w-willlake-Projects-DESI-MultitracerBAO/db282ac8-ac12-4352-9440-1f2f53df60c1/scratchpad/hlo_{label}.txt", "w").write(txt)
