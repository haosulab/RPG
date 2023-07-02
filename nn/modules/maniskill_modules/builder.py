import inspect

BACKBONES = {}

def is_str(x):
    return isinstance(x, str)


def register_all(registry, local_dict, filter=None):
    from torch import nn
    for k, v in local_dict.items():
        if inspect.isclass(v) and issubclass(v, nn.Module):
            #if prefix is None or i.__name__.startswith(prefix):
            if filter is None or filter(k, v):
                registry[k] = v

        
def register_backbone(name=None):
    def wrap_cls(cls):
        xx = name or cls.__name__
        BACKBONES[xx] = cls
        return cls
    return wrap_cls

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError('`cfg` or `default_args` must contain the key "type", but got {cfg}\n{default_args}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {list(registry.keys())} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    
    return obj_cls(**args)