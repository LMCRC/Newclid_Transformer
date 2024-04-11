from __future__ import annotations
from typing import TYPE_CHECKING

from geosolver.pretty import map_symbol

if TYPE_CHECKING:
    from geosolver.problem import Definition


def translate_constrained_to_constructive(
    point: str, name: str, args: list[str]
) -> tuple[str, list[str]]:
    """Translate a predicate from constraint-based to construction-based.

    Args:
      point: str: name of the new point
      name: str: name of the predicate, e.g., perp, para, etc.
      args: list[str]: list of predicate args.

    Returns:
      (name, args): translated to constructive predicate.
    """
    if name in ["T", "perp"]:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return "on_dia", [a, b, d]
        return "on_tline", [a, b, c, d]

    elif name in ["P", "para"]:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        return "on_pline", [a, b, c, d]

    elif name in ["D", "cong"]:
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return "on_bline", [a, b, d]
        if b in [c, d]:
            if b == d:
                c, d = d, c
            return "on_circle", [a, b, d]
        return "eqdistance", [a, b, c, d]

    elif name in ["C", "coll"]:
        a, b, c = args
        if point == b:
            a, b = b, a
        if point == c:
            a, b, c = c, a, b
        return "on_line", [a, b, c]

    elif name in ["^", "eqangle"]:
        a, b, c, d, e, f = args

        if point in [d, e, f]:
            a, b, c, d, e, f = d, e, f, a, b, c

        x, b, y, c, d = b, c, e, d, f
        if point == b:
            a, b, c, d = b, a, d, c

        if point == d and x == y:  # x p x b = x c x p
            return "angle_bisector", [point, b, x, c]

        if point == x:
            return "eqangle3", [x, a, b, y, c, d]

        return "on_aline", [a, x, b, c, y, d]

    elif name in ["cyclic", "O"]:
        a, b, c = [x for x in args if x != point]
        return "on_circum", [point, a, b, c]

    return name, args


def check_valid_args(name: str, args: list[str]) -> bool:
    """Check whether a predicate is grammarically correct.

    Args:
      name: str: name of the predicate
      args: list[str]: args of the predicate

    Returns:
      bool: whether the predicate arg count is valid.
    """
    if name == "perp":
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b}) < 2:
            return False
        if len({c, d}) < 2:
            return False
    elif name == "para":
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b, c, d}) < 4:
            return False
    elif name == "cong":
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b}) < 2:
            return False
        if len({c, d}) < 2:
            return False
    elif name == "coll":
        if len(args) != 3:
            return False
        a, b, c = args
        if len({a, b, c}) < 3:
            return False
    elif name == "cyclic":
        if len(args) != 4:
            return False
        a, b, c, d = args
        if len({a, b, c, d}) < 4:
            return False
    elif name == "eqangle":
        if len(args) != 8:
            return False
        a, b, c, d, e, f, g, h = args
        if len({a, b, c, d}) < 3:
            return False
        if len({e, f, g, h}) < 3:
            return False
    return True


def try_translate_constrained_to_construct(
    string: str, existing_points: list[str], defs: dict[str, "Definition"]
) -> str:
    """Whether a string of aux construction can be constructed.

    Args:
      string: str: the string describing aux construction.
      g: gh.Graph: the current proof state.

    Returns:
      str: whether this construction is valid. If not, starts with "ERROR:".
    """
    if string[-1] != ";":
        return "ERROR: must end with ;"

    head, prem_str = string.split(" : ")
    point = head.strip()

    if len(point) != 1 or point == " ":
        return f"ERROR: invalid point name {point}"

    if point in existing_points:
        return f"ERROR: point {point} already exists."

    prem_toks = prem_str.split()[:-1]  # remove the EOS ' ;'
    prems = [[]]

    for i, tok in enumerate(prem_toks):
        if tok.isdigit():
            if i < len(prem_toks) - 1:
                prems.append([])
        else:
            prems[-1].append(tok)

    if len(prems) > 2:
        return "ERROR: there cannot be more than two predicates."

    clause_txt = point + " = "
    constructions = []

    for prem in prems:
        name, *args = prem

        if point not in args:
            return f"ERROR: {point} not found in predicate args."

        if not check_valid_args(map_symbol(name), args):
            return "ERROR: Invalid predicate " + name + " " + " ".join(args)

        for a in args:
            if a != point and a not in existing_points:
                return f"ERROR: point {a} does not exist."

        try:
            name, args = translate_constrained_to_constructive(point, name, args)
        except ValueError:
            return "ERROR: Invalid predicate " + name + " " + " ".join(args)

        if name == "on_aline":
            if args.count(point) > 1:
                return f"ERROR: on_aline involves twice {point}"

        constructions += [name + " " + " ".join(args)]

    clause_txt += ", ".join(constructions)
    return clause_txt
