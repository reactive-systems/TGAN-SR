"""Implements the rich pattern concatenation generation process for random LTL formulas"""

# values that influence the probabilities during generation are flagged in the following code with # probability parameter

import random
from functools import reduce
import math

import numpy as np

import tgan_sr.utils.ltl_parser as lp



class SpecPatternGen:
    """Stateful object for the generation of a single formula consisting of a concatenation of richly instantiated patterns"""
    def __init__(self, aps, ts):
        """
        aps: atomic proposition (variable) names
        ts: tree size (min,max) tuple
        """
        self.aps = aps
        self.target_length = int(random.random() * (ts[1] - ts[0]) + ts[0]) # sample target length uniformly at random
        self.params = {}
        self.params['var_bind_base'] = np.interp(self.target_length, [1, 10, 100], [0.8, 0.6, 0.2]) # base probability of re-choosing existing variables. scales with target length. # probability parameter
        self.var_collection = set()
        self._var_collection = set()


    def run(self):
        """return a new formula. does not fully reset object state. for a completely unrelated formula, build a new object."""
        current_length = 0
        current_clauses = []
        while (current_length - self.target_length) < 0:
            if random.random() < 0.4: # probability parameter
                pattern = self.ground()
            else:
                pattern = self.pattern()
            pattern = lp.ltl_formula(pattern)
            self.distort_ops(pattern)
            current_clauses.append(pattern)
            current_length += pattern.size()
        return reduce(lp.F_AND, current_clauses)


    def pattern(self):
        """sample a specification pattern and a scope and instantiate them with subformulas"""
        pattern_kind = np.random.choice(['absence', 'universality', 'existence', 'precedence', 'response'])
        scope_kind = np.random.choice(['globally', 'before', 'after', 'between', 'afteruntil'], p=[1/2, 1/8, 1/8, 1/8, 1/8]) # probability parameter

        if pattern_kind == 'absence':
            if scope_kind == 'globally':
                template = 'G (! {P})'
            elif scope_kind == 'before':
                template = 'F {R} -> (!{P} U {R})'
            elif scope_kind == 'after':
                template = 'G ({Q} -> G (!{P}))'
            elif scope_kind == 'between':
                template = 'G (({Q} & !{R} & F{R}) -> (!{P} U {R}))'
            elif scope_kind == 'afteruntil':
                template = 'G ({Q} & !{R} -> (!{P} W {R}))'
        
        elif pattern_kind == 'existence':
            if scope_kind == 'globally':
                template = 'F ({P})'
            elif scope_kind == 'before':
                template = '!{R} W ({P} & ! {R})'
            elif scope_kind == 'after':
                template = 'G (!{Q}) | F ({Q} & F {P})'
            elif scope_kind == 'between':
                template = 'G ({Q} & ! {R} -> (!{R} W ({P} & !{R})))'
            elif scope_kind == 'afteruntil':
                template = 'G ({Q} & !{R} -> (!{R} U ({P} & !{R})))'
        
        elif pattern_kind == 'universality':
            if scope_kind == 'globally':
                template = 'G ({P})'
            elif scope_kind == 'before':
                template = 'F {R} -> ({P} U {R})'
            elif scope_kind == 'after':
                template = 'G ({Q} -> G {P})'
            elif scope_kind == 'between':
                template = 'G (({Q} & !{R} & F {R}) -> ({P} U {R}))'
            elif scope_kind == 'afteruntil':
                template = 'G ({Q} & !{R} -> ({P} W {R}))'
        
        elif pattern_kind == 'precedence':
            if scope_kind == 'globally':
                template = '!{P} W {S}'
            elif scope_kind == 'before':
                template = 'F {R} -> (!{P} U ({S} | {R}))'
            elif scope_kind == 'after':
                template = 'G ! {Q} | F ({Q} & (!{P} W {S}))'
            elif scope_kind == 'between':
                template = 'G (({Q} & !{R} & F {R}) -> (!{P} U ({S} | {R})))'
            elif scope_kind == 'afteruntil':
                template = 'G ({Q} & !{R} -> (!{P} W ({S} | {R})))'
        
        elif pattern_kind == 'response':
            if scope_kind == 'globally':
                template = 'G ({P} -> F {S})'
            elif scope_kind == 'before':
                template = 'F {R} -> ({P} -> (!{R} U ({S} & !{R}))) U {R}'
            elif scope_kind == 'after':
                template = 'G ({Q} -> G ({P} -> F {S}))'
            elif scope_kind == 'between':
                template = 'G (({Q} & !{R} & F {R}) -> ({P} -> (!{R} U ({S} & !{R}))) U {R})'
            elif scope_kind == 'afteruntil':
                template = 'G ({Q} & !{R} -> (({P} -> (!{R} U ({S} & !{R}))) W {R}))'

        instantiate_dict = {}
        if '{P}' in template:
            instantiate_dict['P'] = self.subformula('fancy')
        if '{Q}' in template:
            instantiate_dict['Q'] = self.subformula('broad') # scope event
        if '{R}' in template:
            instantiate_dict['R'] = self.subformula('broad') # scope event
        if '{S}' in template:
            instantiate_dict['S'] = self.subformula('fancy')
        instantiated = template.format(**instantiate_dict)
        return instantiated


    def ground(self):
        """generate a random grounding instantiated with a subformula"""
        subformula = self.subformula('narrow')
        p = random.random()
        if p < 0.1: # probability parameter
            return 'G F ' + subformula
        elif p < 0.2: # probability parameter
            return 'G ' + subformula
        elif p < 0.3: # probability parameter
            return 'F G' + subformula
        elif p < 0.4: # probability parameter
            return 'F ' + subformula
        else:
            num_nexts = int(abs(np.random.normal(0, 3))) # doubles the normal distribution to positive side
            return 'X'*num_nexts + subformula


    def subformula(self, kind='fancy', tlen=None):
        """generate a random subformula. different kinds supported. wrapper around recursive function"""
        res = '(' + self._subformula(tlen=tlen, kind=kind)[0] + ')'
        self.update_vars()
        return res

    def _subformula(self, kind='fancy', depth=0, tlen=None):
        """generate a random subformula (recursive implementation). different kinds supported
        kind : fancy, broad or narrow. Determines operator distribution
        depth : affects operator distribution if on top level or not
        tlen : target length of the formula
        """
        # dp: distribution parameter for normal distribution (mean, stddev)
        if tlen is None:
            tlen = 0
            if kind == 'fancy':
                dp = (2, 3) # probability parameter
            elif kind == 'broad':
                dp = (2, 2) # probability parameter
            elif kind == 'narrow':
                dp = (2, 2) # probability parameter
            while tlen <= 0:
                tlen = int(np.random.normal(*dp))
        assert tlen > 0
        if tlen == 1:
            return self.var(), 1

        ops = ['&', '|', '->', '<->', '!', 'X', 'G', 'F', 'U', 'W']
        if kind == 'fancy':
            ps  = [  6,   2,    2,     2,   5,   4, 0.5, 0.5, 0.5, 0.5] # probability parameters
            if depth == 0:
                ps = [q * 3 if i in [2, 3] else q for (i, q) in enumerate(ps)] # probability parameters
        elif kind == 'broad':
            ps  = [  2,   6,    4,     2,   3,   3, 0.5, 0.5, 0.5, 0.5] # probability parameters
        elif kind == 'narrow':
            ps = [  6,   1,    1,    1,   5,   3, 0.5, 0.5, 0.5, 0.5] # probability parameters

        if tlen == 2:
            ps = [q if i in [4, 5, 6, 7] else 0 for (i, q) in enumerate(ps)] # eliminate binary ops
    
        op = np.random.choice(ops, p=np.array(ps)/sum(ps))
        n, _ = lp.token_dict_spot[op]

        if n == 2:
            lchild, llen = self._subformula(depth=depth+1, tlen=(tlen-1)//2, kind=kind)
            rchild, rlen = self._subformula(depth=depth+1, tlen=(tlen-1-llen), kind=kind)
            return '(' + lchild + ') ' + op + ' (' + rchild + ')', llen + rlen + 1
        elif n == 1:
            child, clen = self._subformula(depth=depth+1, tlen=(tlen-1), kind=kind)
            return op + '(' + child + ')', clen + 1
        else:
            raise ValueError


    def var(self):
        """Samples a new variable. Prefers variables already contained in the formula so far (scales with number of existing variables)"""
        bind_prop = self.params['var_bind_base'] * math.sqrt(len(self.var_collection)) # probability parameter

        if random.random() < bind_prop:
            var = random.sample(self.var_collection, 1)[0]
        else:
            var = random.sample(self.aps, 1)[0]
            self._var_collection.add(var)
        return var


    def update_vars(self):
        self.var_collection = self._var_collection.copy()


    def distort_ops(self, formula):
        """replaces operators by random ones of matching arity"""
        for q in formula:
            if random.random() < 0.01: # probability parameter
                if isinstance(q, lp.LTLFormulaBinaryOp):
                    q.type_ = random.sample(lp.binary_ops, 1)[0] # TODO: clean handling of release & Co.
                elif isinstance(q, lp.LTLFormulaUnaryOp):
                    q.type_ = random.sample(lp.unary_ops, 1)[0]



def arity(token):
    for n, t in lp.token_dict_spot.items():
        if t == token:
            return n
    raise ValueError('illegal token')


def FL(f):
    """get fully-parenthesized string representation of formula"""
    return '(' + f.to_str('spot', full_parens=True) + ')'