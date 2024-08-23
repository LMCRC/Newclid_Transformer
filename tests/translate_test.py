# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for alphageometry.py."""

import pytest

from alphageo.translate import (
    setup_str_from_problem,
    translate_constrained_to_constructive,
)
from geosolver.api import GeometricSolverBuilder
from geosolver.formulations.problem import ProblemJGEX


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (("d", "T", tuple("addb")), "on_dia d b a"),
        (("d", "T", tuple("adbc")), "on_tline d a b c"),
        (("d", "P", tuple("bcda")), "on_pline d a b c"),
        (("d", "D", tuple("bdcd")), "on_bline d c b"),
        (("d", "D", tuple("bdcb")), "on_circle d b c"),
        (("d", "D", tuple("bacd")), "eqdistance d c b a"),
        (("d", "C", tuple("bad")), "on_line d b a"),
        (("d", "C", tuple("bad")), "on_line d b a"),
        (("d", "O", tuple("abcd")), "on_circum d a b c"),
    ],
)
def test_translate_constrained_to_constructive(
    test_input: tuple[str, str, tuple[str]], expected: str
):
    actual = translate_constrained_to_constructive(*test_input)
    assert actual == expected


def test_translate_problem():
    builder = GeometricSolverBuilder(123)
    string = setup_str_from_problem(
        ProblemJGEX.from_text(
            "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
        ).renamed(),
        builder.defs,
    )
    assert string == "{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c"
