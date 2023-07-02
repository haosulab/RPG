# https://www.notion.so/Typed-Dynamics-Graph-316d4c6d9509489ebc97a50e698867a2
import inspect
import copy
import typing
import torch
from torch import nn
from omegaconf import OmegaConf as C
from torch.nn import Module
from .basetypes import Arrow, Type
from .node import Node, nodes_to_types
from .utils import frame_assert


class OptBase(Module):
    def default_config(self) -> C:
        return C.create()


def get_left_value(frame):
    frame = frame[0]
    return inspect.getframeinfo(frame).code_context[0].strip().split("=")[0].strip()


OPID = 0

class Operator(OptBase):
    INFER_SHAPE_BY_FORWARD=False
    arrow: typing.Optional[Arrow] = None # TYPE annotation of the forward funciton
    N_OUTPUT=None

    def __init__(self, *args, name=None, _trace_history=None, **kwargs) -> None:
        super().__init__()
        # TODO: store the lineno for debugging
        self._init_args, self._init_kwargs = args, C.create(kwargs)
        self._name = name or self.__class__.__name__
        self.default_inp_nodes = [Node.from_val(i) for i in args]
        self.clear()

        global OPID
        self._id = OPID
        OPID += 1

        from .utils import exception_with_traceback
        self.call_frame = self.find_caller()
        self.left_value = get_left_value(self.call_frame)
        self._trace_history = ('\n' + _trace_history() if _trace_history is not None else '') + '\n\ninit at\n' + exception_with_traceback(self.call_frame[0]) 

    def find_caller(self):
        # find the caller of this function
        for frame in inspect.stack():
            # hack ..
            if (self.__class__.__name__ in frame[4][0] or 'OPERATORS' in frame[4][0]):
                return frame
        raise ValueError("cannot find the caller of this function")

    def get_trace(self):
        return self._trace_history

    def clear(self):
        self._lazy_init = False
        self._config = None
        self._default_out = None


    def init(self):
        if not self._lazy_init:
            self._lazy_init = True
            self.build_config() # configure it

            try:
                self.main
                has_main = True
            except AttributeError:
                has_main = False

            self.inp_types = nodes_to_types(self.default_inp_nodes)
            if not has_main:
                self.build_modules(*self.inp_types)

    def myassert(self, cond, msg='', errorType=ValueError):
        frame_assert(cond, msg, self.get_trace, errorType)

    # get the output node of the operator based on input nodes ..
    def shadow(self, *input_nodes: typing.List[Node], default=False) -> Node:
        # TODO: for replicate operators.
        self.myassert(default, "left value inference is not implemneted")
        name = self.left_value
        if ',' in name:
            name = '[' + name + ']'

        return Node(parent=self, n_childs=self.get_n_output(*input_nodes), name=name, input_nodes=input_nodes)

    """ type inference """
    def _infer_by_arrow(self, *inp_types):
        _, _, self._out_type = self.arrow.unify(inp_types)
        return self._out_type

    # required for infering the number of output 
    def get_n_output(self, *input_nodes):
        if self.arrow is not None:
            return len(self.arrow.out)
        if self.N_OUTPUT is not None:
            return self.N_OUTPUT
        return 1

    def _type_inference(self, *inp_types):
        raise NotImplementedError

    # required for infering the output type
    def get_output_type_by_input(self, *input_nodes):
        assert not hasattr(self, '_out_type'), "type_inference can only be called once"
        input_types = nodes_to_types(input_nodes)

        if self.INFER_SHAPE_BY_FORWARD:
            self.init() # init the module so that we can infer the output type ..
            shapes = [arg.sample() if isinstance(arg, Type) else arg for arg in input_types]
            output = self.forward(*shapes)
            _out_type = self._get_type_from_output(output, *input_types)
        elif self.arrow is not None:
            _out_type = self._infer_by_arrow(*input_types)
        else:
            assert hasattr(self, '_type_inference'), f"please either override the type_inference function or set the arrow of the class {self.__class__}"
            _out_type = self._type_inference(*input_types)
        return _out_type

    # wrappers
    def get_output(self) -> Node: # out type when input are feed ..
        # TODO: if possible, extract the default name from the line calling the method
        if self._default_out is None:
            self._default_out = self.shadow(*self.default_inp_nodes, default=True)
        return self._default_out

    def __iter__(self):
        return iter(self.get_output())


    """ calling, this is not for the graph construct """
    def __call__(self, *args, **kwargs):
        self.init()

        #TODO: this only checks for the default input nodes ..
        for a, b in zip(self.default_inp_nodes, args):
            a = a.get_type()
            if isinstance(a, Type) and not a.instance(b):
                info = '\n' + str(self)
                info = info.replace('\n', '\n' + '>' * 10)
                from .utils import tensor2error
                frame_assert(False, f"input {tensor2error(b)} does not match the required input type {a} for {info}", self.get_trace, TypeError)
        out = super().__call__(*args, **kwargs)
        # TODO: check output type
        out_type = self.get_output().get_type()
        assert out_type.instance(out), f"output {out} does not match the required output type {out_type}"
        return out

        
    """ config system """
    @property
    def config(self):
        if not hasattr(self, '_config') or self._config is None:
            self.build_config()
        return self._config

    @property
    def pretty_config(self):
        return C.to_yaml(self.config)

    @classmethod
    def _new_config(cls)->C:
        return C.create()

    @classmethod
    def default_config(cls) -> C:
        return C.merge(
            super().default_config(cls),
            cls._new_config()
        )

    def forward(self, *args, **kwargs):
        assert self.main is not None, "please either override this function or set the module of the class"
        try:
            return self.main(*args, **kwargs)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            # print(self.call_frame)
            
            frame_assert(False,
                         f"error in forward function of {self.__class__}:\n{e} with\n {str(tb)}",
                         self.get_trace,
                         e.__class__)
            

    def build_config(self) -> C:
        self._config = C.merge(self.default_config(), self._init_kwargs)


    """ build and handle modules """
    def build_modules(self, *args):
        self.main = None

    def parameters(self, recurse: bool = True):
        self.init()
        return super().parameters(recurse)

    def to(self, *args, **kwargs):
        self.init()
        return super().to(*args, **kwargs)


    """ code for manage computation graph """
    def reconfig(self, **kwargs):
        if self._lazy_init:
            import logging
            logging.warning(f"reconfiguring a module {self._name} that has already been initialized, this is not recommended")
        self._init_kwargs = C.merge(self._init_kwargs, C.create(kwargs))
        self.clear()
        

    def compile(self, *args, **kwargs):
        return self.get_output().compile(*args, **kwargs)
        

    """ utilities """
    def __str__(self) -> str:
        #out = super().__str__()
        self.init()
        out = torch.nn.Module.__str__(self)
        return out + f" -> {self.get_output().get_type()}"

    def _get_type_from_output(self, output, *args):
        inp = args[0]
        out_shape = inp.new(*inp.batch_shape(), *output.shape[-inp.data_dims:])
        return out_shape

    def __hash__(self) -> int:
        return hash(f'THISISANOPWITHID:{self._id}')

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self):
        raise NotImplementedError("deepcopy is not supported for now")