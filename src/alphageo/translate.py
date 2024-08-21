from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Collection
from geosolver.tools import atomize
import geosolver.predicates as predicates
from geosolver.predicates import Predicate
from geosolver.problem import ProblemJGEX
from geosolver.definition.definition import DefinitionJGEX

if TYPE_CHECKING:
    ...

MAP_SYMBOL: dict[str, type[Predicate]] = {
    "T": predicates.Perp,
    "P": predicates.Para,
    "D": predicates.Cong,
    "S": predicates.SimtriClock,
    "I": predicates.Circumcenter,
    "M": predicates.MidPoint,
    "O": predicates.Cyclic,
    "C": predicates.Coll,
    "^": predicates.EqAngle,
    "/": predicates.EqRatio,
    "%": predicates.EqRatio,
    "=": predicates.ContriClock,
}


def translate_constrained_to_constructive(
    point: str, name: str, args: tuple[str]
) -> str:
    """Translate a predicate from constraint-based to construction-based.

    Args:
      point: str: name of the new point
      name: str: name of the predicate, e.g., perp, para, etc.
      args: list[str]: list of predicate args.

    Returns:
      (name, args): translated to constructive predicate.
    """
    return MAP_SYMBOL[name].to_constructive(point, args)


def try_translate_constrained_to_construct(
    s: str, existing_points: Collection[str]
) -> str:
    """Whether a string of aux construction can be constructed.

    Args:
      string: str: the string describing aux construction.
      g: gh.Graph: the current proof state.

    Returns:
      str: whether this construction is valid. If not, starts with "ERROR:".
    """
    if s[-1] != ";":
        return "ERROR: must end with ';'"

    if ":" not in s:
        return "ERROR: must contain ':'"

    point, prem_str = atomize(s, ":")

    if not (point.isalpha() and len(point) == 1):
        return f"ERROR: invalid point name {point}"

    if point in existing_points:
        return f"ERROR: point {point} already exists."

    prem_toks = atomize(prem_str)[:-1]  # remove the EOS ' ;'
    prems: list[list[str]] = [[]]

    for i, tok in enumerate(prem_toks):
        if tok.isdigit():
            if i < len(prem_toks) - 1:
                prems.append([])
        else:
            prems[-1].append(tok)

    if len(prems) > 2:
        return "ERROR: there cannot be more than two predicates."

    constructives: list[str] = []

    for prem in prems:
        try:
            name = prem[0]
            args = tuple(prem[1:])
        except ValueError:
            return f"ERROR: {prem} is not in the form of name + args."

        if point not in args:
            return f"ERROR: {point} not found in predicate args."

        try:
            pred = MAP_SYMBOL[name]
            for a in args:
                if a != point and a not in existing_points:
                    return f"ERROR: point {a} does not exist."
            constructive = pred.to_constructive(point, args)
        except Exception:
            return f"ERROR: predicate {name} {' '.join(args)} failed to be transformed into a constructive"

        constructives.append(constructive)

    clause_txt = point + " = " + ", ".join(constructives)
    return clause_txt


def pretty2r(a: str, b: str, c: str, d: str) -> str:
    if b in (c, d):
        a, b = b, a

    if a == d:
        c, d = d, c

    return f"{a} {b} {c} {d}"


def pretty2a(a: str, b: str, c: str, d: str) -> str:
    if b in (c, d):
        a, b = b, a

    if a == d:
        c, d = d, c

    return f"{a} {b} {c} {d}"


def pretty(s: tuple[str, ...]) -> str:
    """Pretty formating a predicate string."""
    name, *args = s
    if name == "aconst":
        a, b, c, d, y = args
        return f"^ {pretty2a(a, b, c, d)} {y}"
    if name == "rconst":
        a, b, c, d, y = args
        return f"/ {pretty2r(a, b, c, d)} {y}"
    if name == "coll":
        return "C " + " ".join(args)
    if name == "collx":
        return "X " + " ".join(args)
    if name == "cyclic":
        return "O " + " ".join(args)
    if name in ["midp", "midpoint"]:
        x, a, b = args
        return f"M {x} {a} {b}"
    if name == "eqangle":
        a, b, c, d, e, f, g, h = args
        return f"^ {pretty2a(a, b, c, d)} {pretty2a(e, f, g, h)}"
    if name == "eqratio":
        a, b, c, d, e, f, g, h = args
        return f"/ {pretty2r(a, b, c, d)} {pretty2r(e, f, g, h)}"
    if name == "eqratio3":
        a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
        return f"S {o} {a} {b} {o} {c} {d}"
    if name == "cong":
        a, b, c, d = args
        return f"D {a} {b} {c} {d}"
    if name == "perp":
        if len(args) == 2:  # this is algebraic derivation.
            ab, cd = args  # ab = 'd( ... )'
            return f"T {ab} {cd}"
        a, b, c, d = args
        return f"T {a} {b} {c} {d}"
    if name == "para":
        if len(args) == 2:  # this is algebraic derivation.
            ab, cd = args  # ab = 'd( ... )'
            return f"P {ab} {cd}"
        a, b, c, d = args
        return f"P {a} {b} {c} {d}"
    if name in ["simtri2", "simtri", "simtri*"]:
        a, b, c, x, y, z = args
        return f"S {a} {b} {c} {x} {y} {z}"
    if name in ["contri2", "contri", "contri*"]:
        a, b, c, x, y, z = args
        return f"= {a} {b} {c} {x} {y} {z}"
    if name == "circle":
        o, a, b, c = args
        return f"I {o} {a} {b} {c}"
    if name == "foot":
        a, b, c, d = args
        return f"F {a} {b} {c} {d}"
    return " ".join(s)


def hashed_txt(name: str, args: list[str]) -> tuple[str, ...]:
    """Return a tuple unique to name and args upto arg permutation equivariant."""

    if name in ["const", "aconst", "rconst"]:
        a, b, c, d, y = args
        a, b = sorted([a, b])
        c, d = sorted([c, d])
        return name, a, b, c, d, y

    if name in ["npara", "nperp", "para", "cong", "perp", "collx"]:
        a, b, c, d = args

        a, b = sorted([a, b])
        c, d = sorted([c, d])
        (a, b), (c, d) = sorted([(a, b), (c, d)])

        return (name, a, b, c, d)

    if name in ["midp", "midpoint"]:
        a, b, c = args
        b, c = sorted([b, c])
        return (name, a, b, c)

    if name in ["coll", "cyclic", "ncoll", "diff", "triangle"]:
        return (name,) + tuple(sorted(list(set(args))))

    if name == "circle":
        x, a, b, c = args
        return (name, x) + tuple(sorted([a, b, c]))

    if name in ["eqangle", "eqratio", "eqangle6", "eqratio6"]:
        a, b, c, d, e, f, g, h = args
        a, b = sorted([a, b])
        c, d = sorted([c, d])
        e, f = sorted([e, f])
        g, h = sorted([g, h])
        if tuple(sorted([a, b, e, f])) > tuple(sorted([c, d, g, h])):
            a, b, e, f, c, d, g, h = c, d, g, h, a, b, e, f
        if (a, b, c, d) > (e, f, g, h):
            a, b, c, d, e, f, g, h = e, f, g, h, a, b, c, d

        if name == "eqangle6":
            name = "eqangle"
        if name == "eqratio6":
            name = "eqratio"
        return (name,) + (a, b, c, d, e, f, g, h)

    if name in ["contri", "simtri", "simtri2", "contri2", "contri*", "simtri*"]:
        a, b, c, x, y, z = args
        (a, x), (b, y), (c, z) = sorted([(a, x), (b, y), (c, z)], key=sorted)
        (a, b, c), (x, y, z) = sorted([(a, b, c), (x, y, z)], key=sorted)
        return (name, a, b, c, x, y, z)

    if name in ["eqratio3"]:
        a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
        (a, c), (b, d) = sorted([(a, c), (b, d)], key=sorted)
        (a, b), (c, d) = sorted([(a, b), (c, d)], key=sorted)
        return (name, a, b, c, d, o, o)

    if name in ["sameside", "s_angle"]:
        return (name,) + tuple(args)

    raise ValueError(f"Not recognize {name} to hash.")


def _gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x


def simplify(n: int, d: int) -> tuple[int, int]:
    g = _gcd(n, d)
    return (n // g, d // g)


def compare_fn(s: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    return (s, pretty(s))


def sort_deps(s: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return sorted(s, key=compare_fn)


def setup_str_from_problem(
    problem: ProblemJGEX, definitions: dict[str, DefinitionJGEX]
) -> str:
    """Construct the <theorem_premises> string from Problem object."""
    ref = 0

    string: list[str] = []
    for clause in problem.constructions:
        group: dict[str, tuple[str, ...]] = {}
        p2deps: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
        for c in clause.sentences:
            cdef = definitions[c[0]]

            if len(c[1:]) != len(cdef.declare[1:]):
                assert len(c[1:]) + len(clause.points) == len(cdef.declare[1:])
                c = c[0:1] + clause.points + c[1:]

            mapping = dict(zip(cdef.declare[1:], c[1:]))
            for points, bs in cdef.basics:
                points = tuple([mapping[x] for x in points])
                for p in points:
                    group[p] = points

                for b in bs:
                    args = [mapping[a] for a in b[1:]]
                    name = b[0]
                    if b[0] in ["s_angle", "aconst"]:
                        x, y, z, v = args
                        name = "aconst"
                        v = int(v)

                        if v < 0:
                            v = -v
                            x, z = z, x

                        m, n = simplify(int(v), 180)
                        args = [y, z, y, x, f"{m}pi/{n}"]

                    p2deps[points].append(hashed_txt(name, args))

        for k, v in p2deps.items():
            p2deps[k] = sort_deps(v)

        points: tuple[str, ...] = clause.points
        while points:
            p = points[0]
            gr = group[p]
            points = tuple(x for x in points if x not in gr)

            deps_str: list[str] = []
            for dep in p2deps[gr]:
                ref_str = "{:02}".format(ref)
                dep_str = pretty(dep)

                if dep[0] == "aconst":
                    m, n = map(int, dep[-1].split("pi/"))
                    mn = f"{m}. pi / {n}."
                    dep_str = " ".join(dep_str.split()[:-1] + [mn])

                deps_str.append(dep_str + " " + ref_str)
                ref += 1

            string.append(" ".join(gr) + " : " + " ".join(deps_str))

    res = "{S} " + " ; ".join([s.strip() for s in string])
    goal = problem.goals[0]
    res += " ? " + pretty(goal)
    return res
