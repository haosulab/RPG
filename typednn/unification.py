# https://github.com/hzaskywalker/TaskAnnotator/blob/main/llm/pl/unification.py
#from .types import TypeInferenceFailure, Arrow, List, Type, Tuple
"""
TODO:
1. support inherit or union types
"""
import typing 
from .basetypes import TupleType, Type, VariableArgs


class TypeInferenceFailure(Exception):
    """Error in type inference"""


def map_type(tp: Type, f):
    if len(tp.children()) > 0:
        return tp.reinit([map_type(i, f) for i in tp.children()])
    else:
        return f(tp)

def check_occurs(a, b):
    if str(a) == str(b):
        return True
    for i in b.children():
        if check_occurs(a, i):
            return True
    return False


def contain_many(X):
    s = sum([i.match_many() for i in X.children()])
    assert s <= 1, "can not have two variable arguments."
    return s > 0
    

def unify(tpA: Type, tpB: Type, query: Type):

    if isinstance(tpA, list):
        tpA = TupleType(*tpA)
    if isinstance(tpB, list):
        tpB = TupleType(*tpB)
    if isinstance(query, list):
        query = TupleType(*query)

    tpB = tpB.update_name(lambda x: x + '\'')
    query = query.update_name(lambda x: x + '\'')

    pa = {} # map str to type in the end ..
    def findp(a: Type):
        if a.is_type_variable:
            s = str(a)
            if s not in pa:
                pa[s] = a
            if pa[s] != a:
                pa[s] = findp(pa[s])
            return pa[s]
        else:
            return a

    def resolve(a, b, dir):
        # if dir is 1, we want to unify a into b
        a = findp(a)
        b = findp(b)
        if a.is_type_variable:
            if str(a) != str(b):
                if check_occurs(a, b):
                    raise TypeInferenceFailure("Recursive type ..")
                pa[str(a)] = b
        elif b.is_type_variable:
            resolve(b, a, dir=1-dir)
        else:
            # a, b both are not type variable 
            if len(a.children()) == 0:
                # two normal types mismatch
                #if str(a) != str(b):
                #    raise TypeInferenceFailure(f"type {a} != type {b}.")
                if dir == 1:
                    a, b = b, a
                if str(a) != str(b):
                    raise TypeInferenceFailure(f"type {a} != type {b}.")
                return
            if contain_many(b):
                a, b = b, a
                dir = 1 - dir
                
            # now we assume a must contain ... or there is no ...
            if not contain_many(a):
                # no ..., directly match two lists
                a_children, b_children = a.children(), b.children()
                if len(a_children) != len(b_children):
                    raise TypeInferenceFailure(f"type {a} has a different number of children with type {b}.")
                for x, y in zip(a_children, b_children):
                    resolve(x, y, dir)
            else:
                C, D = a.children(), b.children()
                def match_prefix(C, D):
                    idx = 0
                    while True:
                        if C[idx].match_many() or D[idx].match_many():
                            C = C[idx:]
                            D = D[idx:]
                            break
                        resolve(C[idx], D[idx], dir)
                        idx += 1
                    return C, D

                # we now remove the common prefix or suffix until meet a ... or the end
                C, D = match_prefix(C, D)
                C, D = match_prefix(C[::-1], D[::-1]); C, D = C[::-1], D[::-1]

                # make C start with ...
                if len(D) > 0 and D[0].match_many():
                    C, D = D, C
                    dir = 1 - dir
                assert C[0].match_many()

                def resolve_many(arg_types, types, dir):
                    if arg_types.base_type is not None:
                        for i in types:
                            resolve(arg_types.base_type, i, dir)
                    resolve(arg_types, TupleType(*types), dir)

                if C[0].match_many() and D[0].match_many():
                    # the most difficult case
                    # ... blabla
                    if len(C) > 1:
                        if len(D) != 1:
                            raise TypeInferenceFailure
                        C, D = D, C
                        dir = 1 - dir
                    # C is ...; D is ... blabla
                    resolve_many(C[0], D, dir)
                else:
                    if len(C) != 1:
                        raise TypeInferenceFailure
                    # associate C[0] to the remaining element of D
                    resolve_many(C[0], D, dir)



    resolve(tpA, tpB, 0)

    allocator = {}
    TID = 0

    def substitute(x: Type):
        if not x.polymorphism:
            return x

        if not x.is_type_variable and len(x.children()) > 0:
            # can't be type variable
            out = []
            for i in x.children():
                if isinstance(i, VariableArgs):
                    y = substitute(i)
                    if isinstance(y, TupleType):
                        out += y.children()
                    else:
                        out.append(y)
                else:
                    out.append(substitute(i))
            return x.reinit(*out)

        assert x.is_type_variable
        p = findp(x)
        if p.is_type_variable:
            if p._type_name not in allocator:
                nonlocal TID
                allocator[p._type_name] = p.reinit(f'\'T{TID}', *[substitute(i) for i in p.children()])
                TID += 1
        else:
            return p
        out = allocator[p._type_name]
        return out

    tpA = substitute(tpA)
    tpB = substitute(tpB)
    query = substitute(query)
    return tpA, tpB, query