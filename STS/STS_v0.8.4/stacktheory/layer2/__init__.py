# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Layer 2 compilers, adapters, and representations.

Layer 2 is where the library becomes convenient to use.
It standardises tensor encodings, provides conversions to and from Stack Theory
objects, and adds interoperability formats.

Layer 2 does not change any Layer 1 semantics.
It only changes how objects are represented and moved between systems.
"""

from .adapters import (
    statement_from_mask_tensor,
    statement_to_mask_tensor,
    vocabulary_from_tensor,
    vocabulary_to_tensor,
)
from .ternary import (
    clause_tensor_to_clause,
    clause_to_clause_tensor,
    cnf_tensor_to_cnf,
    cnf_to_cnf_tensor,
    dnf_tensor_to_dnf,
    dnf_to_dnf_tensor,
    validate_ternary_matrix,
)
from .pretty import (
    cnf_to_str,
    dnf_to_str,
    parse_cnf,
    parse_dnf,
)
from .dimacs import (
    cnf_to_dimacs,
    cnf_from_dimacs,
)
from .sympy_bridge import (
    cnf_to_sympy,
    dnf_to_sympy,
    sympy_to_cnf,
    sympy_to_dnf,
)
from .bitpack import (
    pack_bool_tensor_uint64,
    unpack_bool_tensor_uint64,
    popcount_uint64,
)

from .wordbitset import (
    WordBitset,
    popcount_words,
)

from .sat import (
    SATResult,
    SATHeuristicResult,
    LNSConfig,
    sat_solve_cnf,
    sat_solve_cnf_lns,
    sat_is_satisfiable,
)

__all__ = [
    'statement_from_mask_tensor',
    'statement_to_mask_tensor',
    'vocabulary_from_tensor',
    'vocabulary_to_tensor',
    'validate_ternary_matrix',
    'clause_tensor_to_clause',
    'clause_to_clause_tensor',
    'cnf_tensor_to_cnf',
    'cnf_to_cnf_tensor',
    'dnf_tensor_to_dnf',
    'dnf_to_dnf_tensor',
    'cnf_to_str',
    'dnf_to_str',
    'parse_cnf',
    'parse_dnf',
    'cnf_to_dimacs',
    'cnf_from_dimacs',
    'cnf_to_sympy',
    'dnf_to_sympy',
    'sympy_to_cnf',
    'sympy_to_dnf',
    'pack_bool_tensor_uint64',
    'unpack_bool_tensor_uint64',
    'popcount_uint64',
    'WordBitset',
    'popcount_words',
    'SATResult',
    'SATHeuristicResult',
    'LNSConfig',
    'sat_solve_cnf',
    'sat_solve_cnf_lns',
    'sat_is_satisfiable',
]
