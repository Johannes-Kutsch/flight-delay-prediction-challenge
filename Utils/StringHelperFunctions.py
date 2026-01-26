import re

def strip_ansi_codes(string):
    return re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]').sub('', string)

def camel_to_snake(string):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()