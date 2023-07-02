import typing
import numpy as np
from ..basetypes import Type, TupleType, VariableArgs


class UIntType(Type):
    def __init__(self, a):
        self.a = a
    def __str__(self):
        return str(self.a)

    def instance(self, value):
        return isinstance(value, int) and value == self.a 

    def sample(self):
        return self.a

    def __repr__(self):
        return "Unit(" + str(self.a) + ")"

    def __int__(self):
        return int(self.a)
        

class SizeType(TupleType):
    def __init__(self, *size: typing.List[Type]):
        _size = []
        for i in size:
            if isinstance(i, str):
                if i.startswith('...'):
                    i = VariableArgs(i)
                else:
                    i = Type(i)
            _size.append(i)
        size = _size

        self.dot = None # (int, ..., int) something like this ..
        self.size = []
        for i in size:
            if isinstance(i, int):
                i = UIntType(i)
            else:
                assert i.match_many() or isinstance(i, UIntType) or i.is_type_variable, f"{i} of {type(i)}"
                
            self.size.append(i)
        super().__init__(*self.size)

    def dims(self):
        if self.dot is None:
            return len(self.size)
        return None

    def sample(self):
        #return super().sample()
        outs = []
        for i in self.size:
            if isinstance(i, UIntType):
                outs.append(i.sample())
            else:
                if isinstance(i, VariableArgs):
                    n = np.random.randint(3)
                else:
                    n = 1
                for i in range(n):
                    outs.append(np.random.randint(1, 10))
        return tuple(outs)
                
    def __str__(self):
        return '(' + ', '.join(map(str, self.size)) + ')'

    def __repr__(self):
        return 'Size'+self.__str__()

    def __getitem__(self, index):
        out = self.size[index]
        if isinstance(out, UIntType):
            return int(out)
        elif isinstance(out, list):
            out = SizeType(*out)
        return out

    def __add__(self, other):
        if isinstance(other, SizeType):
            return SizeType(*(self.size + other.size))
        return SizeType(*(self.size + [other]))

    def __iter__(self):
        return iter(self.size)

    def total(self):
        strings = []
        total = 1
        for i in self.size:
            if isinstance(i, UIntType):
                total *= int(i)
            else:
                if isinstance(i, VariableArgs):
                    strings.append('prod(' + i._type_name + ')')
                else:
                    strings.append(i._type_name)

        if len(strings) > 0:
            if total > 1:
                strings = [str(total)] + strings
            if len(strings) == 1:
                return Type(strings[0])
            return Type('$' + '(' + '*'.join(strings) + ')')
        else:
            return UIntType(total)

    def as_int(self):
        return [int(i) for i in self.size]

            

def test():
    size = SizeType('...', 'N', 'M')
    print(size)
    print(size.total())
            

if __name__ == '__main__':
    test()