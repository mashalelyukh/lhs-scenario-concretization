import re
from lark import Lark, Transformer

#expr: param "in" "[" value ".." value "]"

dsl_range_grammar = r"""
    start: in_expr | colon_expr | call_expr | enum_in_expr | enum_colon_expr | enum_call_expr

    in_expr: param "in" "[" value RANGE_OP value "]"
    colon_expr: param ":" "[" value RANGE_OP value "]"
    call_expr: param "(" "[" value RANGE_OP value "]" ("," extras)? ")"
    extras: /[^)]+/

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
        return self._build_numeric("in", param, min_val, max_val)

    def colon_expr(self, items):
        param, min_val, _, max_val = items
        return self._build_numeric("colon", param, min_val, max_val)

    #def call_expr(self, items):
     #   param, min_val, _, max_val = items
      #  return self._build_numeric("call", param, min_val, max_val)

    def call_expr(self, items):
        param, min_val, _, max_val, *rest = items
        extras = str(rest[0]) if rest else ""
        return {
            "type": "call",
            "param": param,
            "min": float(min_val[0]),
            "max": float(max_val[0]),
            "unit": min_val[1] if min_val[1] else "",
            "extras": extras
        }

    def enum_in_expr(self, items):
        return self._build_enum("in", *items)

    def enum_colon_expr(self, items):
        return self._build_enum("colon", *items)

    def enum_call_expr(self, items):
        return self._build_enum("call", *items)

    def _build_numeric(self, kind, param, min_val, max_val):
        return {
            "type": kind,
            "param": param,
            "min": float(min_val[0]),
            "max": float(max_val[0]),
            "unit": min_val[1] if min_val[1] else ""
        }

    def _build_enum(self, kind, param, enum_values):
        return {
            "type": kind,
            "param": param,
            "values": enum_values,
            "enum": True
        }

    def param(self, items):
        return str(items[0])

    def value(self, items):
        number = items[0]
        unit = items[1] if len(items) > 1 else None
        return (number, str(unit) if unit else None)

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
    #parser = Lark(dsl_range_grammar, start="expr")
    parser = Lark(dsl_range_grammar, start="start")
    transformer = RangeTransformer()
    #results = []
    numerical_parameters = {}
    enum_parameters = {}

    param_types = {}
    for match in re.finditer(r"(?P<param>\w+)\s*:\s*(?P<type>\w+)", content):
        param_types[match.group("param")] = match.group("type")

    patterns = [
        ("in", r"\b\w+\s+in\s+\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        ("colon", r"\b\w+\s*:\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]"),
        #("call", r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]\s*\)")
        ("call", r"\b\w+\s*\(\s*\[\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\.\.\s*-?\d+(?:\.\d+)?(?:[a-zA-Z]+)?\s*\]\s*(?:,\s*[^)]*)?\)")

    ]

    enum_patterns = [
        ("enum_in", r'\b\w+\s+in\s+\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]'),
        ("enum_colon", r'\b\w+\s*:\s*\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]'),
        ("enum_call", r'\b\w+\s*\(\s*\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]\s*\)')
    ]

    for line in content.splitlines():
        for label, pattern in patterns + enum_patterns:
            for match in re.finditer(pattern, line):
                expr = match.group().strip()
                try:
                    tree = parser.parse(expr)
                    ast = transformer.transform(tree)

                    # Only add metadata if ast is a dict
                    if isinstance(ast, dict):
                        param_name = ast["param"]
                        ast["type_annotation"] = param_types.get(param_name)
                        ast["original"] = expr
                        ast["line"] = line
                        ast["type"] = label

                        if "enum" in ast:
                            enum_parameters.setdefault(param_name, []).append(ast)
                            #enum_parameters[param_name] = ast
                        else:
                            if ast["type_annotation"] is None:
                            #???????????????????????????????????????????????????????????????????????
                                if ast["min"].is_integer() and ast["max"].is_integer():
                                   ast["type_annotation"] = "int"
                                else:
                                    ast["type_annotation"] = "float"
                            #???????????????????????????????????????????????????????????????????????

                                #ast["type_annotation"] = "int"

                                #if float(ast["min"]).is_integer() and float(ast["max"]).is_integer():
                                 #   ast["type_annotation"] = "int"
                                #else:
                                 #   ast["type_annotation"] = "float"

                            if ast["type_annotation"] in ("int", "uint"):
                                ast["min"] = int(ast["min"])
                                ast["max"] = int(ast["max"])
                            elif ast["type_annotation"] == "float":
                                ast["min"] = float(ast["min"])
                                ast["max"] = float(ast["max"])

                            #old: numerical_parameters[param_name] = ast
                            numerical_parameters.setdefault(param_name, []).append(ast)

                            #ast["original"] = expr
                            #ast["line"] = line
                            #ast["type"] = label
                            #results.append(ast)
                    else:
                        print(f"Warning: parsed but transformer did not return a dict for: {expr}")
                    #tree = parser.parse(expr)
                    #ast = transformer.transform(tree)
                    #ast["original"] = expr
                    #ast["line"] = line
                    #results.append(ast)
                except Exception as e:
                    print(f"Failed to parse [{label}] pattern: {expr}\n{e}")
    #return results
    return numerical_parameters, enum_parameters


def extract_parameters(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    #numerical_parameters = {}
    #enum_parameters = {}

    numerical_parameters, enum_parameters = extract_range_dsl_statements(content)

    #for param_info in extract_range_dsl_statements(content):
     #   numerical_parameters[param_info['param']] = param_info

    return numerical_parameters, enum_parameters
