import re
from lark import Lark, Transformer

#expr: param "in" "[" value ".." value "]"

dsl_range_grammar = r"""
    start: in_expr | colon_expr | call_expr

    in_expr: param "in" "[" value RANGE_OP value "]"
    colon_expr: param ":" "[" value RANGE_OP value "]"
    call_expr: param "(" "[" value RANGE_OP value "]" ")"

    param: CNAME
    value: NUMBER UNIT?

    UNIT: /[a-zA-Z]+/
    RANGE_OP: ".."
    NUMBER: /-?\d+(\.\d+)?/

    %import common.CNAME
    %import common.WS
    %ignore WS
"""



class RangeTransformer(Transformer):
    def start(self, items):
        return items[0]

    def in_expr(self, items):
        param, min_val, _, max_val = items
        return self._build("in", param, min_val, max_val)

    def colon_expr(self, items):
        param, min_val, _, max_val = items
        return self._build("colon", param, min_val, max_val)

    def call_expr(self, items):
        param, min_val, _, max_val = items
        return self._build("call", param, min_val, max_val)

    def _build(self, kind, param, min_val, max_val):
        return {
            "type": kind,
            "param": param,
            "min": float(min_val[0]),
            "max": float(max_val[0]),
            "unit": min_val[1] if min_val[1] else ""
        }

    def param(self, items):
        return str(items[0])

    def value(self, items):
        number = items[0]
        unit = items[1] if len(items) > 1 else None
        return (number, str(unit) if unit else None)


def extract_range_dsl_statements(content):
    #parser = Lark(dsl_range_grammar, start="expr")
    parser = Lark(dsl_range_grammar, start="start")
    transformer = RangeTransformer()
    results = []

    patterns = [
        ("in", r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        ("colon", r"\b\w+\s*:\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        ("call", r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]\s*\)")
    ]

    #patterns = [
        #r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]", #param in [*..*]
        #r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\.\.\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\]",  # param in [..] second
        #r"\b\w+\s*:\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]", #param: [] main



        #r"\b\w+\s*:\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]", #hz
        #r"\b\w+\s*:\s*\[\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\.\.\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\]",  # param: [..] hz
        #r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\.\.\s*-?\d+(?:\.\d+)?[a-zA-Z]*\s*\]\s*\)"  # param([..])

    #]

    #range_expr_regex = r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"

    for line in content.splitlines():
        for label, pattern in patterns:
            for match in re.finditer(pattern, line):
                expr = match.group().strip()
                try:
                    tree = parser.parse(expr)
                    ast = transformer.transform(tree)

                    # Only add metadata if ast is a dict
                    if isinstance(ast, dict):
                        ast["original"] = expr
                        ast["line"] = line
                        results.append(ast)
                    else:
                        print(f"Warning: parsed but transformer did not return a dict for: {expr}")
                    #tree = parser.parse(expr)
                    #ast = transformer.transform(tree)
                    #ast["original"] = expr
                    #ast["line"] = line
                    #results.append(ast)
                except Exception as e:
                    print(f"Failed to parse [{label}] pattern: {expr}\n{e}")
    return results


def extract_parameters(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    numerical_parameters = {}
    enum_parameters = {}

    for param_info in extract_range_dsl_statements(content):
        numerical_parameters[param_info['param']] = param_info

    return numerical_parameters, enum_parameters
