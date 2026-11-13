# Axion

**A**utodifferentiable, **X**LA-compiled, **I**mplicit, **O**n-device **N**umerics:
a differentiable, GPU-native finite element framework written in
[JAX](https://github.com/jax-ml/jax), demonstrated on phase-field fracture.

**Website:** coming soon

This repository contains the framework and every numerical example of

> W. Wang, M. Krishnan U., L. Graham-Brady, S. Goswami,
> *Axion: A differentiable, GPU-native finite element framework, demonstrated on
> phase-field fracture.*

The element kernels, the Newton and Krylov iterations, the bound-constrained
updates, the staggered coupling of fields and the loop over load increments are
expressed as one program that is just-in-time compiled and executed on the GPU:

```
scan over load increments [ while staggered [ while Newton [ while CG ] ] ]
```

* **One element definition per field.** The user writes the weak form at a
  quadrature point. The element residual and its consistent tangent come from
  the same kernel; the tangent is obtained by forward-mode automatic
  differentiation.
* **Element-by-element operators.** Global operators are applied from the stored
  element tangents (gather, local product, scatter-add) without assembling a
  sparse matrix; linear systems are solved by Jacobi-preconditioned CG.
* **Fixed array shapes, compiled control flow.** Bound constraints are enforced
  by projection, adaptively refined meshes live in fixed-capacity buffers, and
  a cell-based domain decomposition runs the same program on several GPUs.
* **Differentiable.** Forward-mode derivatives of the discrete solution with
  respect to model parameters are obtained by differentiating the same program.

`axion/` is self-contained. It depends on JAX and NumPy (meshio for output) and
does not use or import any other finite element library.

## Installation

```bash
conda env create -f environment.yml
conda activate axion
```

`environment.yml` installs the CPU build of JAX. On a CUDA machine install the
matching GPU wheel instead, for example `pip install -U "jax[cuda12]"`. The
results of the paper were obtained with JAX 0.10.2 and CUDA 12 in double
precision.

The examples add the repository root to `sys.path` themselves, so no install
step is needed; `pip install -e .` makes `import axion` available elsewhere.

## A first example

```python
import jax.numpy as jnp
import axion
from axion import DirichletBC, Field, rectangle_mesh

E, nu = 210e3, 0.3
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

class Elasticity(Field):
    def flux(self):                       # paired with grad v
        def stress(grad_u):
            eps = 0.5 * (grad_u + grad_u.T)
            return lam * jnp.trace(eps) * jnp.eye(2) + 2 * mu * eps
        return stress

mesh = rectangle_mesh(40, 40)             # unit square, QUAD4
bottom = lambda x: jnp.isclose(x[1], 0.0)
top = lambda x: jnp.isclose(x[1], 1.0)
u = Elasticity(mesh, ncomp=2, dirichlet=[DirichletBC(bottom, 0), DirichletBC(bottom, 1),
                                         DirichletBC(top, 1, 1e-3)])

fixed, values = u.dirichlet_dofs, u.dirichlet_values
z0 = jnp.zeros(u.num_dofs).at[fixed].set(values)
R0, V0 = axion.assemble(u, z0)            # residual and element tangents (by AD)
newton = axion.make_newton(u, tol=1e-8)   # compiled Newton + Jacobi-CG
z = newton(z0, R0.at[fixed].set(0.0), V0, (), fixed)
axion.write_vtu("elastic.vtu", mesh, point_data={"u": z.reshape(-1, 2)})
```

A weak form is declared through three hooks of `Field`, each returning a plain
JAX function of the values at one quadrature point:

| hook | signature | paired with |
|---|---|---|
| `flux()` | `fn(grad_z, *theta) -> (ncomp, dim)` | `grad v` in the cell |
| `source()` | `fn(z, x, *theta) -> (ncomp,)` | `v` in the cell |
| `boundary_sources()` | list of `fn(z, x, *theta_b) -> (ncomp,)` | `v` on each boundary |

`theta` collects the data held fixed while the field is solved (material
parameters, internal variables, the other fields), sampled at the quadrature
points. Coupled problems are sets of fields solved in turn; see the examples.

## Repository layout

```
axion/
  elements.py      TRI3 / QUAD4 / TET4 / HEX8 shape functions and quadrature
  mesh.py          mesh container, structured mesh generators
  field.py         one field equation: weak form -> element residuals, tangents by
                   forward-mode AD, residual assembly, operator action
  spectral.py      isotropic tensor functions (spectral split) with a derivative
                   that is valid for repeated eigenvalues
  solvers.py       Newton + Jacobi-CG, projected Jacobi, active-set Newton
  distributed.py   multi-GPU execution by cell decomposition
  io.py            VTU output
examples/          one folder per numerical example of the paper
tests/             checks of the core on tiny meshes (run on a CPU in seconds)
```

## Numerical examples

| Folder | Paper | Problem | Size [DoF] |
|---|---|---|---|
| `01_sent_2d` | Sec. 4.1.1 | 2D single-edge-notched tension, AT2; uniform and adaptive meshes | 3.08e4 / 1.13e4 |
| `02_sent_3d` | Sec. 4.1.2 | 3D single-edge-notched tension, AT2, HEX8 | 1.46e6 |
| `03_parallel_scaling` | Sec. 4.1.3 | strong and weak scaling of the 3D benchmark on up to 4 GPUs | 1.46e6 - 5.6e6 |
| `04_derivative_verification` | Sec. 4.1.4 | Taylor test of the tangents; forward-mode sensitivities vs. finite differences | 6.8e4 |
| `05_hydraulic_fracture` | Sec. 4.2.1 | three-field hydro-poroelastic fracture, two configurations | 3.80e5 |
| `06_premolar` | Sec. 4.2.2 | 3D premolar, six materials, contact, PF-CZM, TET4 | 6.87e5 |
| `07_koyna_dam` | Sec. 4.2.3 | Koyna gravity dam, uniform mesh, one and two GPUs | 9.6e7 |

Every folder has a README with the settings of the paper, the command to run
and the files that are written. Settings are read from environment variables
(`PFF_*`), so a small test is one line, for example

```bash
cd examples/02_sent_3d
PFF_NX=40 PFF_NY=40 PFF_NZ=4 PFF_NSTEPS=10 python sent_3d.py
```

The solvers are meant for a GPU. On a CPU they run, but the compilation of the
load-loop program alone can take tens of minutes.

## Tests

```bash
python tests/test_core.py          # or: pytest tests
python -m axion.distributed        # multi-GPU kernels on 4 emulated CPU devices
```

## Citation

```bibtex
@article{axion,
  title  = {Axion: A differentiable, GPU-native finite element framework,
            demonstrated on phase-field fracture},
  author = {Wang, Wei and Krishnan U., Meenu and Graham-Brady, Lori and Goswami, Somdatta},
  year   = {2026}
}
```

## License

Axion is released under the [MIT License](LICENSE).
