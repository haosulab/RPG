# computation node
# TODO: structured node (DictNode)

import abc
import typing
import copy
from .basetypes import Type


def nodes_to_types(nodes):
    from .node import Node
    nodes: typing.List[Node]
    return [i.get_type() for i in nodes]


# global NODEID
NODEID = 0

class NodeBase(abc.ABC):
    @classmethod
    def from_val(cls, val):
        if isinstance(val, NodeBase):
            return val

        from .operator import Operator
        if isinstance(val, Operator):
            module: Operator = val
            return module.get_output()
        elif isinstance(val, Type):
            return InputNode(val)
        else:
            return ValNode(val)

    @abc.abstractmethod
    def get_type(self):
        pass

    def __init__(self, name=None) -> None:
        super().__init__()
        self._name = name

        global NODEID
        self._id = NODEID
        NODEID += 1

    def get_parents(self):
        # find all nodes that are connected to this node
        return []

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.get_type()})'

    def __hash__(self) -> int:
        return hash(f'THISISANODEWITHID:{self._id}')

    def __copy__(self):
        raise NotImplementedError("copy is not supported for Nodes")

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        #node = super().__deepcopy__(memo)
        node = copy.deepcopy(self)
        self.__deepcopy__ = deepcopy_method
        node.__deepcopy__ = deepcopy_method

        global NODEID
        node._id = NODEID
        NODEID += 1
        return node

    @abc.abstractmethod
    def evaluate(self, context):
        pass


class InputNode(NodeBase):
    # TODO: remove input node ...
    def __init__(self, type, **kwargs) -> None:
        super().__init__(**kwargs)
        self._type = type

    def get_type(self):
        return self._type

    def evaluate(self, context):
        return context[self]


class ValNode(NodeBase):
    def __init__(self, val, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val = val

    def get_type(self):
        return self.val

    def evaluate(self, context):
        return self.val


class Node(NodeBase):
    def __init__(
        self,
        parent=None,
        input_nodes=None,
        index=None,
        n_childs=0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        from .operator import Operator

        if isinstance(parent, Operator):
            assert input_nodes is not None, "input_types should not be None"
        elif isinstance(parent, NodeBase):
            assert index is not None, "index should not be None"
        else:
            raise NotImplementedError("parent should be an Operator or TypeNode")

        self._type = None
        self._parent = parent
        self._index = index
        self._n_childs = n_childs

        self.input_nodes = input_nodes

    def get_type(self):
        # lazy evaluation
        if self._type is None:
            from .operator import Operator
            if isinstance(self._parent, Operator):
                self._type = self._parent.get_output_type_by_input(*self.input_nodes)
            else:
                self._type = self._parent.get_type()[self._index]
        return self._type

    def get_parents(self):
        if isinstance(self._parent, NodeBase):
            return [self._parent]
        else:
            return self.input_nodes

    def __iter__(self):
        for i in range(self._n_childs):
            if ',' in self._name:
                name = str(self._name)[1:-1].split(',')[i]
            else:
                name = self._name + f'.{i}'
            yield Node(self, index=i, name=name)


    def compile(self, *args, **kwargs):
        from .compiler import compile
        return compile(self, *args, **kwargs)

    def evaluate(self, context):
        if isinstance(self._parent, NodeBase):
            return self._parent.evaluate(context)[self._index]
        else:
            for i in self.input_nodes:
                if i not in context:
                    context[i] = i.evaluate(context)
            return self._parent(*[context[i] for i in self.input_nodes])



if __name__ == '__main__':
    node = ValNode(1)
    import copy
    node2 = copy.deepcopy(node)
    node2 = copy.deepcopy(node)
    assert hash(node2) != hash(node)
    print(node._id)
    print(node2._id)
    print(NODEID)