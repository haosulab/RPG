# TODO: add hash for TYPES if necessary
# we can also support infering the auxiliary data information from the input data information; for example, the shape and dtypes.
import typing

def match_list(A, B: typing.List["Type"]):
    if len(A) != len(B):
        return False
    for a, b in zip(A, B):
        if not b.instance(a):
            return False
    return True

def iterable(x):
    return isinstance(x, typing.Iterable)


class Type:
    def __init__(self, type_name) -> None:
        self._type_name = type_name
        assert self.__class__ == Type, "Type should not be instantiated directly."

    def reinit(self, *children):
        # create new types based on the provided chidren
        return self.__class__(*children, **self._get_extra_info())

    def _get_extra_info(self):
        return {}

    @property
    def is_type_variable(self):
        return hasattr(self, '_type_name')

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._type_name

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def update_name(self, fn) -> "Type":
        if not self.polymorphism:
            return self
        args = []
        if self.is_type_variable:
            args.append(fn(self._type_name))
        return self.reinit(*args, *[i.update_name(fn) for i in self.children()])

    def children(self) -> typing.Tuple["Type"]:
        return ()
    
    def match_many(self):
        return False

    def sample(self):
        raise NotImplementedError("sample is not implemented for type %s" % self.__class__.__name__)

    @property
    def polymorphism(self):
        #TODO: accelerate this ..
        if hasattr(self, '_polymorphism'):
            return self._polymorphism
        self._polymorphism = True
        if self.is_type_variable:
            return True
        for i in self.children():
            if i.polymorphism:
                return True
        self._polymorphism = False
        return False

    def instance(self, x):
        return True


class TupleType(Type):
    # let's not consider named tuple for now ..
    def __init__(self, *args: typing.List[Type]) -> None:
        self.elements = []
        self.dot = None
        for idx, i in enumerate(args):
            if i.match_many():
                assert self.dot is None, NotImplementedError("only one ellipsis is allowed.")
                self.dot = idx
            self.elements.append(i)

    def __str__(self):
        return f"({', '.join(str(e) for e in self.children())})"

    def children(self):
        return tuple(self.elements) # + tuple(self.elements_kwargs.values())

    def instance(self, inps):
        #assert isinstance(inps, tuple) or isinstance(inps, list)
        if not iterable(inps):
            return False
        inps = list(inps)

        if self.dot is None and len(inps) != len(self.elements):
            return False
        if self.dot is not None and len(inps) < len(self.elements)-1:
            return False

        if self.dot is None:
            return match_list(inps, self.elements)
        else:
            l = self.dot
            r = len(self.elements) - l - 1
            if l > 0 and not match_list(inps[:l], self.elements[:l]):
                return False
            if r > 0 and not match_list(inps[-r:], self.elements[-r:]):
                return False
            return self.elements[l].instance(inps[l:-r])
    
    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TupleType(*self.elements[idx])
        return self.elements[idx]



class ListType(Type): # sequence of data type, add T before the batch
    def __init__(self, base_type: Type) -> None:
        self.base_type = base_type

    def __str__(self):
        return f"List({self.base_type})"

    def children(self):
        return (self.base_type,)

    def instance(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            return False
        for i in x:
            if not self.base_type.instance(i):
                return False
        return True


class VariableArgs(Type): # is in fact the ListType with unknown length
    # something that can match arbitrary number of types
    def __init__(self, type_name, based_type: typing.Optional["Type"]=None):
        self._type_name = type_name
        self.base_type = based_type

    def match_many(self):
        return True

    def __str__(self):
        if self.base_type is None:
            return self._type_name + "*"
        return self._type_name + ":" + str(self.base_type) + "*"

    def instance(self, x):
        if not iterable(x):
            return False
        if self.base_type is None:
            return True
        for i in x:
            if not self.base_type.instance(i):
                return False
        return True

    def children(self):
        if self.base_type is not None:
            return (self.base_type,)
        return ()
    

class PType(Type):
    # probablistic distribution of the base_type
    def __init__(self, base_type) -> None:
        raise NotImplementedError


class DataType(Type):
    # data_cls could be anything ..
    def __init__(self, data_cls, type_name=None):
        self.data_cls = data_cls
        self.type_name = type_name or self.data_cls.__name__

    def __str__(self):
        #return self.data_cls.__name__
        return self.type_name

    def instance(self, x):
        return isinstance(x, self.data_cls)

    def children(self):
        return ()

        
class UnionType(Type):
    def __init__(self, *types) -> None:
        self.types = tuple(types)

    def __str__(self):
        return f"Union({', '.join(str(e) for e in self.children())})"

    def instance(self, x):
        return any(i.instance(x) for i in self.types)

    def children(self) -> typing.Tuple["Type"]:
        return self.types

        
class Arrow(Type):
    def __init__(self, *args) -> None:
        self.args = args[:-1]
        self.out = args[-1]

    def __str__(self):
        return '->'.join(str(e) for e in self.children())

    def children(self) -> typing.Tuple["Type"]:
        return list(self.args) + [self.out]

    def unify(self, *args):
        from .unification import unify
        return unify(TupleType(*args), TupleType(*self.args), self.out)

    def test_unify(self, gt, *args):
        from .unification import TypeInferenceFailure
        print("testing unify", self)
        print("INPUT:")
        for i in args:
            print(" " + str(i))
        print("Output:")
        if gt != 'error':
            output = str(self.unify(*args)[-1])
            assert output == gt, "unify failed: " + output + " != " + gt
            print("unify succeed! ", output + ' == ' + gt)
            print("\n\n")
        else:
            try:
                self.unify(*args)
                assert False, "unify should fail!"
            except TypeInferenceFailure:
                print("unify failed as expected.")
                print("\n\n")

    def instance(self, x):
        raise NotImplementedError("Arrow is not a simple type, it's a function type.")