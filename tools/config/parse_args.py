import inspect
from functools import wraps
import sys

from .configurable import Configurable, match_inputs, merge_a_into_b_builder, merge_inputs, CN

EXTRA_PARSER = []

def _parse_args(default_cfg='', parser=None):
    import argparse
    if parser is None:
        parser = argparse.ArgumentParser(description='Training')

    parser.add_argument(
        '--cfg',
        dest='config_file',
        default=default_cfg,
        help='path to config file',
        type=str,
    )

    parser.add_argument(
        '--var',
        default=None,
        help='experiment variant to run',
        type=str,
    )

    parser.add_argument(
        '--output_cfg',
        action="store_true"
    )
    parser.add_argument(
        '--save_cfg',
        default=None,
        help='path to save config file',
    )

    for i in EXTRA_PARSER:
        i(parser)

    args, unknown = parser.parse_known_args()
    args.opts = unknown
    return args


def inherit(key, variants):
    assert key in variants, f"key {key} not found in variants definition."
    import copy
    var = copy.copy(variants[key])
    if '_inherit' in var:
        father = inherit(var.pop('_inherit'), variants)
        father.set_new_allowed(True)
        now = CN(var)
        out = merge_a_into_b_builder(now, father)

        return out
    return CN(var)


def parse_args(default_cfg_path='', parser=None, parse_prefix=None, strict=False, _update=None, _variants=None):
    # remember if we use this to decorate a configurable
    # it works with the initial __init__

    def parse_decorator(init):
        signature = inspect.signature(init)

        @wraps(init)  # this helps to preserve the signature.. very interesting..
        def wrapper(self, *args, **kwargs):
            args, cfg, kwargs = match_inputs(signature, *args, **kwargs)
            opts = _parse_args(default_cfg_path, parser=parser)
            opt_cfg = opts.config_file if hasattr(opts, 'config_file') and opts.config_file != '' and opts.config_file is not None else None

            for i in opts.opts[::2]:
                if i[:2] != '--' and i != '-f': # for jupyter notebook
                    raise KeyError(f"Please add -- before opts.. for key {i}")
            opt_kwargs = {a[2:]: b for a, b in zip(opts.opts[::2], opts.opts[1::2]) if a!='-f'}

            if parse_prefix is not None:
                prefix_len = len(parse_prefix) + 1
                opt_kwargs = {
                    a[prefix_len:]: b for a, b in opt_kwargs.items()
                    if a.startswith(parse_prefix + '.')
                }
            # print(parse_prefix, opt_kwargs)

            opt_cfg = merge_inputs(opt_cfg, **opt_kwargs)
            cfg: CN
            cfg.set_new_allowed(True)
            cfg._exp_name = opts.var
            tmp = CN(cfg.copy())

            
            update = _update
            if opts.var is not None or (_variants is not None and 'default' in _variants):
                if _variants is None:
                    raise KeyError(f"Experiment management configs not in cfg")
                if opts.var is None:
                    opts.var = _variants['default']

                if opts.var not in _variants:
                    raise KeyError(f"Exp {opts.var} not in {_variants.keys()}")
                #base = CN(_variants[opts.var])
                
                base = inherit(opts.var, _variants)
                if update is not None:
                    merge_a_into_b_builder(CN(update), base)
                update = base

            if update is not None:
                merge_a_into_b_builder(CN(update), cfg)

            try:
                merge_a_into_b_builder(opt_cfg, cfg)
            except KeyError as e:
                cfg = tmp
                cfg.set_new_allowed(True) # this seems very dangerous
                merge_a_into_b_builder(opt_cfg, cfg)
                print("ERROR while adding arguments before!!!", e)

                if strict:
                    raise e.with_traceback()

            cfg.freeze()

            if hasattr(opts, 'save_cfg') and opts.save_cfg is not None:
                with open(opts.save_cfg, 'w') as f:
                    f.write(str(Configurable.purge(cfg, level=10, strict=False)))

            if hasattr(opts, 'output_cfg') and opts.output_cfg:
                print(Configurable.purge(cfg, level=10, strict=False))
                # sys.exit(0)

            for i in kwargs:
                kwargs[i] = cfg[i]
            
            return init(self, *args, cfg=cfg, **kwargs)

        return wrapper

    return parse_decorator
