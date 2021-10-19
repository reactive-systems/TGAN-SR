"""helper wrapper module for solving LTL formulas with aalta"""

import os.path as path
import subprocess
import re

from tgan_sr.utils import ltl_parser

DEFAULT_BINARY_PATH = "bin"


def solve_ltl(formula_obj, timeout=None, binary_path=DEFAULT_BINARY_PATH):
    formula_str = formula_obj.rewrite(ltl_parser.Token.WEAKUNTIL).to_str('spot', spacing='all ops', full_parens=True) # unambiguous form
    try:
        sat = aalta_sat(formula_str, timeout=timeout, binary_path=binary_path)
    except RuntimeError:
        sat = None
    return sat, None, {}


def aalta_sat(formula: str, timeout=None, binary_path=DEFAULT_BINARY_PATH, mute=False) -> bool:
    """Calls aalta to check if the provided formula is satisfiable"""
    full_aalta_path = path.join(binary_path, 'aalta')
    try:
        res = subprocess.run([full_aalta_path], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        if not mute:
            print("!! aalta threw an error: " + e.stderr)
            print('for input formula', formula)
        return None
    m = re.fullmatch('please input the formula:\n((?:un)?sat)\n', res.stdout, re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("Regular expression for aalta output did not match. Output: '" + res.stdout + "'")
    res_sat = m.groups()[0]
    assert res_sat == 'sat' or res_sat == 'unsat'
    return res_sat == 'sat'
