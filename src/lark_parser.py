import re
from lark import Lark, Transformer
import math

# expr: param "in" "[" value ".." value "]"

dsl_range_grammar = r"""
    start: in_expr | colon_expr | call_expr | enum_in_expr | enum_colon_expr | enum_call_expr

    in_expr: param "in" "[" value RANGE_OP value "]"
    colon_expr: param ":" "[" value RANGE_OP value "]"
    call_expr: param "(" "[" value RANGE_OP value "]" ("," EXTRAS)? ")"
    EXTRAS: /[^)]+/

    enum_in_expr: param "in" "[" enum_values "]"
    enum_colon_expr: param ":" "[" enum_values "]"
    enum_call_expr: param "(" "[" enum_values "]" ")"
    
    
    enum_values: ESCAPED_STRING ("," ESCAPED_STRING)*

    param: CNAME
    value: NUMBER UNIT?

    RANGE_OP: ".."
    NUMBER: /-?\d+(\.\d+)?/
    UNIT: /[a-zA-Z]+/
    
    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""


class RangeTransformer(Transformer):
    def start(self, items):
        return items[0]

    def in_expr(self, items):
        param, min_val, _, max_val = items
        return self.build_numeric("in", param, min_val, max_val)

    def colon_expr(self, items):
        param, min_val, _, max_val = items
        return self.build_numeric("colon", param, min_val, max_val)

    def call_expr(self, items):
        param, min_val, _, max_val, *rest = items
        extras = str(rest[0]) if rest else ""
        return {
            "type": "call",
            "param": param,
            #"min": float(min_val[0]),
            #"max": float(max_val[0]),
            "min": min_val,
            "max": max_val,
            "unit": min_val[1] if min_val[1] else "",
            "extras": extras.strip()
        }

    def enum_in_expr(self, items):
        return self.build_enum("in", *items)

    def enum_colon_expr(self, items):
        return self.build_enum("colon", *items)

    def enum_call_expr(self, items):
        return self.build_enum("call", *items)

    def build_numeric(self, kind, param, min_val, max_val):
        return {
            "type": kind,
            "param": param,
            #"min": float(min_val[0]),
            #"max": float(max_val[0]),
            "min": min_val,
            "max": max_val,
            "unit": min_val[1] if min_val[1] else ""
        }

    def build_enum(self, kind, param, enum_values):
        return {
            "type": kind,
            "param": param,
            "values": enum_values,
            "enum": True
        }

    def param(self, items):
        return str(items[0])

    def value(self, items):
        number_token = str(items[0])
        number = float(number_token)
        is_explicit_float = '.' in number_token
        unit = items[1] if len(items) > 1 else None
        return (number, str(unit) if unit else None, is_explicit_float)

    def enum_values(self, items):
        return [s[1:-1] for s in items]


def extract_param_types(content):
    param_types = {}
    for match in re.finditer(r"(?P<param>\w+)\s*:\s*(?P<type>\w+)", content):
        param = match.group("param")
        param_type = match.group("type")
        param_types[param] = param_type
    return param_types


def extract_range_dsl_statements(content):
    parser = Lark(dsl_range_grammar, start="start")
    transformer = RangeTransformer()
    numerical_parameters = {}
    enum_parameters = {}

    param_types = {}

    for line in content.splitlines():
        m = re.match(r"\s*(?P<param>\w+)\s*:\s*(?P<type>int|uint|float|string)\b", line)
        if m:
            param_types[m.group("param")] = m.group("type")

    patterns = [
        ("in", r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        ("colon", r"\b\w+\s*:\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        # ("call", r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]\s*\)")
        ("call", r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]\s*(?:,"
                 r"\s*[^)]*)?\)")
    ]

    enum_patterns = [
        ("enum_in", r'\b\w+\s+in\s+\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]'),
        ("enum_colon", r'\b\w+\s*:\s*\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]'),
        ("enum_call", r'\b\w+\s*\(\s*\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]\s*\)')
    ]

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue

        code_part = raw_line
        if '#' in raw_line:
            code_part = raw_line.split('#', 1)[0]

        for label, pattern in patterns + enum_patterns:
            # for match in re.finditer(pattern, line):
            for match in re.finditer(pattern, code_part):
                expr = match.group().strip()
                #span = match.span()
                try:
                    tree = parser.parse(expr)
                    ast = transformer.transform(tree)

                    if not isinstance(ast, dict):
                        continue

                    # remember where it came from
                    ast.update({
                        "original": expr,
                        "line": raw_line,
                        "line_number": line_number,
                        "span": match.span(),
                        "type": label,
                    })

                    if ast.get("enum"):
                        ast["type_annotation"] = "string"
                        enum_parameters.setdefault(ast["param"], []).append(ast)
                        continue

                    # validating numeric structure
                    if not ("min" in ast and "max" in ast):
                        raise ValueError(f"Expected numeric fields 'min' and 'max' in AST: {ast}")

                    # figure out declared type (or infer)
                    raw_type = param_types.get(ast["param"])
                    min_val, _, min_is_float = ast["min"]
                    max_val, _, max_is_float = ast["max"]

                    if raw_type in ("int", "uint", "float", "string"):
                        ast["type_annotation"] = raw_type
                    else:
                        if min_is_float or max_is_float:
                            ast["type_annotation"] = "float"
                        elif float(min_val).is_integer() and float(max_val).is_integer():
                            ast["type_annotation"] = "int"
                        else:
                            ast["type_annotation"] = "float"

                    # flatten min/max
                    ast["min"], ast["max"] = float(min_val), float(max_val)

                    # enums just pass through
                    #if ast.get("enum"):
                     #   enum_parameters.setdefault(ast["param"], []).append(ast)
                      #  continue

                    # ASAM’s conversion rules fro numerics
                    ta = ast["type_annotation"]
                    orig_min = float(ast["min"])
                    orig_max = float(ast["max"])

                    if ta in ("int", "uint"):
                        # detect mismatch: float bounds for int/uint, or negative in uint
                        need_adjust = (
                                orig_min != int(orig_min)
                                or orig_max != int(orig_max)
                                or (ta == "uint" and orig_min < 0)
                        )

                        if need_adjust:
                            print(
                                f"WARNING: Provided parameter range for {ast['param']} does not match the datatype")
                            # ceil/floor then clamp to ≥0 if uint
                            new_min = math.ceil(orig_min)
                            new_max = math.floor(orig_max)
                            if ta == "uint":
                                new_min = max(new_min, 0)
                                new_max = max(new_max, 0)
                        else:
                            new_min = int(orig_min)
                            new_max = int(orig_max)

                        ast["min"], ast["max"] = new_min, new_max

                    elif ta == "float":
                        # floats accept int or float implicitly
                        ast["min"], ast["max"] = orig_min, orig_max
                    numerical_parameters.setdefault(ast["param"], []).append(ast)

                except Exception as e:
                    print(f"Failed to parse [{label}] pattern: {expr}\n{e}")
    return numerical_parameters, enum_parameters


def extract_parameters(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    numerical_parameters, enum_parameters = extract_range_dsl_statements(content)

    return numerical_parameters, enum_parameters
