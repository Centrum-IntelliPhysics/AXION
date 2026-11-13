"""Axion: a differentiable, GPU-native finite element framework written in JAX.

A coupled problem is a set of fields. For each field the user writes the weak
form at one quadrature point; Axion generates the element residuals, the
element tangents (by forward-mode automatic differentiation), the
element-by-element operator action and the solvers, all of which can be
compiled into a single program.
"""
import jax

# Phase-field fracture has a stiffness contrast of 1e8 between intact and
# broken material, which single precision cannot resolve.
jax.config.update("jax_enable_x64", True)

from .elements import ReferenceElement, reference_element      # noqa: E402
from .mesh import Mesh, box_mesh, rectangle_mesh               # noqa: E402
from .field import DirichletBC, FaceSet, Field                 # noqa: E402
from .spectral import (eigvalsh_3x3, isotropic_function,       # noqa: E402
                       negative_part, positive_part)
from .solvers import (abs_diagonal, assemble, make_active_set_newton,  # noqa: E402
                      make_newton, make_operator, make_projected_jacobi)
from .io import write_vtu                                      # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "ReferenceElement", "reference_element",
    "Mesh", "box_mesh", "rectangle_mesh",
    "DirichletBC", "FaceSet", "Field",
    "eigvalsh_3x3", "isotropic_function", "negative_part", "positive_part",
    "abs_diagonal", "assemble", "make_active_set_newton", "make_newton", "make_operator",
    "make_projected_jacobi",
    "write_vtu",
]
