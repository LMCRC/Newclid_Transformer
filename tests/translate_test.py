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

from alphageo.translate import translate_constrained_to_constructive


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (("d", "T", list("addb")), ("on_dia", ["d", "b", "a"])),
        (("d", "T", list("adbc")), ("on_tline", ["d", "a", "b", "c"])),
        (("d", "P", list("bcda")), ("on_pline", ["d", "a", "b", "c"])),
        (("d", "D", list("bdcd")), ("on_bline", ["d", "c", "b"])),
        (("d", "D", list("bdcb")), ("on_circle", ["d", "b", "c"])),
        (("d", "D", list("bacd")), ("eqdistance", ["d", "c", "b", "a"])),
        (("d", "C", list("bad")), ("on_line", ["d", "b", "a"])),
        (("d", "C", list("bad")), ("on_line", ["d", "b", "a"])),
        (("d", "O", list("abcd")), ("on_circum", ["d", "a", "b", "c"])),
    ],
)
def test_translate_constrained_to_constructive(test_input, expected):
    actual = translate_constrained_to_constructive(*test_input)
    assert actual == expected
