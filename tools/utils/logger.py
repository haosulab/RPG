import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
# import wandb
from typing import Union, List
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps
MPI = None


class KVWriter(ABC):
    @abstractmethod
    def writekvs(self, kvs):
        pass

    @abstractmethod
    def close(self):
        pass

    def write_image(self, k, v):
        pass

    def write_video(self, k, v):
        pass

    def writemedia(self, kvs):
        import numpy as np
        for k, v in kvs.items():
            v = np.array(v)
            if len(v.shape) == 3:
                self.write_image(k, v)
            elif len(v.shape) == 4:
                self.write_video(k, v)


class SeqWriter(ABC):
    @abstractmethod
    def writeseq(self, seq):
        pass

    @abstractmethod
    def close(self):
        pass


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                    "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = []
        for (key, val) in sorted(kvs.items()):
            if ':std' not in key:
                if hasattr(val, "__float__"):
                    valstr = "%-8.5g" % val
                else:
                    valstr = str(val)
                if key + ':std' in kvs:
                    valstr = "%0.5g" % val + ' +/- ' + "%-8.5g" % (kvs[key + ':std'])

                key2str.append((self._truncate(key), self._truncate(valstr)))

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(lambda kv: len(kv[0]), key2str))
            valwidth = max(map(lambda kv: len(kv[1]), key2str))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str, key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

    def writemedia(self, kvs):
        return super().writemedia(kvs)

    def write_image(self, k, v):
        import cv2
        cv2.imwrite(dir_path(k+'.png'), v[...,::-1]) # convert to opencv2 format

    def write_video(self, k, v):
        #return super().write_video(k, v)
        if k[-4:] != '.mp4':
            k = k+'.mp4'
        animate(v, k)


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

    def writemedia(self, kvs):
        pass


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if hasattr(v, "__float__"):
                v = float(v)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()

    def writemedia(self, kvs):
        pass


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(self.dir)
        self.all_writers = {}

    def _get_writer(self, k):
        if '/' not in k:
            return self.writer, k
        else:
            group, name = k.split('/', 1)
            if group not in self.all_writers:
                from tensorboardX import SummaryWriter
                self.all_writers[group] = SummaryWriter(os.path.join(self.dir, group))
            return self.all_writers[group], name

    def writekvs(self, kvs):
        for k, v in kvs.items():
            w, name = self._get_writer(k)
            w.add_scalar(name, v, self.step)
        self.step += 1
        self.writer.flush()
        for v in self.all_writers.values():
            v.flush()

    def write_image(self, k, v):
        assert '/' not in k, "use _get_writer intead"
        self.writer.add_image(f'image/{k}', v, self.step, dataformats='HWC')

    def write_video(self, k, v):
        import torch
        w, name = self._get_writer(k)
        w.add_video(f'video/{name}', torch.tensor(v).permute(0, 3, 1, 2)[None,:], self.step, fps=10)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        for v in self.all_writers.values():
            v.close()
        self.all_writers = None

        
class WandbOutputFormat(KVWriter):
    def __init__(
        self,
        path,
        config,
        **kwargs
    ) -> None:
        super().__init__()
        self.run = wandb.init(dir=path, reinit=True, config=config, **kwargs)

    def writekvs(self, kvs):
        self.run.log(kvs)

    def close(self):
        self.run.finish()


FORMARTS = {}

def register_format(name):
    def wrapper(cls):
        FORMARTS[name] = cls
        return cls
    return wrapper
 

def make_output_format(format, ev_dir, log_suffix="", **kwargs):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    elif format == "wandb":
        if len(log_suffix) > 0:
            ev_dir = osp.join(ev_dir, log_suffix)
        return WandbOutputFormat(ev_dir, **kwargs)
    elif format in FORMARTS:
        return FORMARTS.get(format)(ev_dir, log_suffix)
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkv_mean_std(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean_std(key, val)



def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def logkvs_mean(d, prefix=None):
    """
    Log a dictionary of key-value pairs with averaging over multiple calls
    """
    for (k, v) in d.items():
        if prefix: k = prefix + k
        logkv_mean(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def dump_media(kvs):
    return get_current().dump_media(kvs)


def getkvs():
    return get_current().name2val


def log(*args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args)



def warn(*args):
    get_current().warn(*args)


def get_dir():
    """
    Get directory that log files are being written to.
    Will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()

def get_run():
    if get_current().use_wandb:
        return get_current().use_wandb.run
    return None


def dir_path(path):
    if isinstance(path, str) and path[0] != '/':
        path = os.path.join(get_dir(), path)
    return path

    
def ifconfigured(f):
    def wrapper(*args, **kwargs):
        if is_configured():
            return f(*args, **kwargs)
        else:
            return
    return wrapper


@ifconfigured
def torch_save(obj, f, best=None):
    L = get_current()
    if L.log_suffix is not None and len(L.log_suffix) == 0:
        if best is not None and not hasattr(L, 'best_torch_score'):
            import numpy as np
            L.best_torch_score = defaultdict(lambda: -np.inf)
        import torch
        torch.save(obj, dir_path(f))
        if best is not None and best > L.best_torch_score[f]:
            L.best_torch_score[f] = best
            torch.save(obj, dir_path('best_' + f))

@ifconfigured
def animate(images, f, *args, wandb_name=None, **kwargs):
    from .utils import animate 
    ani = animate(list(images), dir_path(f), *args, **kwargs)
    if get_current().use_wandb:
        #get_current().use_wandb.run.log({wandb_name or f: wandb.Video(dir_path(f))})
        #get_current().use_wandb.run.log({wandb_name or f: wandb.save(dir_path(f))})
        get_current().use_wandb.run.save(dir_path(f))
        assert not f.endswith('.gif'), "we don't support gif for video now"
        html="""
<video width="320" height="240" controls>
<source src="FILENAME" type="video/TYPE">
</video>"""
        write_html(html.replace('FILENAME', f).replace("TYPE", f.split('.')[-1]), f'{f}.html')
    return ani


@ifconfigured
def save_pcd(xyz, path, color=None):
    import open3d as o3d
    from . import tonumpy
    if not get_current().use_wandb:
        if not isinstance(xyz, o3d.geometry.PointCloud):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(tonumpy(xyz))
        else:
            pcd = xyz
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(tonumpy(color))

        f = path
        o3d.io.write_point_cloud(dir_path(f), pcd)
    else:
        xyz = tonumpy(xyz)
        if color is not None:
            import numpy as np
            xyz = np.concatenate([xyz, tonumpy(color)[:,::-1] * 255], axis=-1)
        get_current().use_wandb.run.log({path: wandb.Object3D(xyz)})


@ifconfigured
def savefig(f, im=None, *, clear=True):
    import matplotlib.pyplot as plt
    if im is not None:
        plt.clf()
        plt.axis('off')
        plt.imshow(im)
    if get_current().use_wandb:
        get_current().use_wandb.run.log({f: plt})
    else:
        plt.tight_layout()
        plt.savefig(dir_path(f))
    if clear:
        plt.clf()


def write_html(txt, f):
    with open(dir_path(f), 'w') as file:
        file.write(txt)
    run = get_run()
    if run is not None:
        # https://api.wandb.ai/files/feyat/fling/pwowngs4/rod.png
        # https://wandb.ai/files/feyat/fling/runs/pwowngs4/rod.png

        url = run.get_url().replace('wandb.ai', 'api.wandb.ai/files').replace('/runs/', '/')
        txt = txt.replace('src=\"', 'src=\"'+url+'/')
        run.log({f: wandb.Html(txt)})


@contextmanager
def profile_kv(scopename, sync_cuda=False):
    if sync_cuda:
        _sync_cuda()
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        if sync_cuda:
            _sync_cuda()
        get_current().name2val[logkey] += time.time() - tstart


def _sync_cuda():
    from torch import cuda

    cuda.synchronize()


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func, name):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with profile_kv(name):
                return func(*args, **kwargs)

        return func_wrapper

    if callable(n):
        return decorator_with_name(n, n.__name__)
    elif isinstance(n, str):
        return partial(decorator_with_name, name=n)
    else:
        raise NotImplementedError(
            "profile should be called as either a bare decorator"
            " or with a string (profiling name of a function) as an argument"
        )


def dump_kwargs(func):
    """
    Prints all keyword-only parameters of a function. Useful to print hyperparameters used.
    Usage:
    @logger.dump_kwargs
    def create_policy(*, hp1, hp2, hp3): ...
    or
    logger.dump_kwargs(ppo.learn)(lr=60e-5, ...)
    """

    def func_wrapper(*args, **kwargs):
        import inspect, textwrap

        sign = inspect.signature(func)
        for k, p in sign.parameters.items():
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                default = "%15s (default)" % str(sign.parameters[k].default)
                get_current().log(
                    "%s.%s: %15s = %s"
                    % (
                        func.__module__,
                        func.__qualname__,
                        k,
                        textwrap.shorten(
                            str(kwargs.get(k, default)),
                            width=70,
                            drop_whitespace=False,
                            placeholder="...",
                        ),
                    )
                )
        return func(*args, **kwargs)

    return func_wrapper


# ================================================================
# Backend
# ================================================================

# Pytorch explainer:
# If you keep a reference to a variable that depends on parameters, you
# keep around the whole computation graph. That causes an unpleasant surprise
# if you were just trying to log a scalar. We could cast to float, but
# that would require a synchronization, and it would be nice if logging
# didn't require the value to be available immediately. Therefore we
# detach the value at the point of logging, and only cast to float when
# dumping to the log file.


def get_current():
    if not is_configured():
        raise Exception("you must call logger.configure() before using logger")
    return Logger.CURRENT


class Logger(object):
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2sqr = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm
        # from .checkpoint import ModelCheckpoint
        # self.checkpoint_saver = None

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        if hasattr(val, "requires_grad"):  # see "pytorch explainer" above
            val = val.detach()
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        assert hasattr(val, "__float__")
        if hasattr(val, "requires_grad"):  # see "pytorch explainer" above
            val = val.detach()
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def logkv_mean_std(self, key, val):
        assert hasattr(val, "__float__")
        if hasattr(val, "requires_grad"):  # see "pytorch explainer" above
            val = val.detach()

        import torch
        import numpy as np
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        oldval, cnt = self.name2val[key], self.name2cnt[key]

        if isinstance(val, np.ndarray):
            val = val.reshape(-1)
            ncnt = cnt + val.shape[0]

            self.name2val[key] = oldval * cnt / ncnt + val.sum() / ncnt
            self.name2sqr[key] = (self.name2sqr[key] * cnt + (val * val).sum()) / ncnt
            self.name2cnt[key] = ncnt
        # raise NotImplementedError("logkv_mean_std not implemented")

        else:
            self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
            self.name2sqr[key] = (self.name2sqr[key] * cnt + (val * val)) / (cnt + 1)
            self.name2cnt[key] = cnt + 1
        # raise NotImplementedError("logkv_mean_std not implemented")

    def dump_media(self, kvs):
        for fmt in self.output_formats:
            fmt.writemedia(kvs)

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
            name2sqr = self.name2sqr
        else:
            d = self.comm.weighted_mean(
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            name2sqr = self.comm.weighted_mean(
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2sqr.items()
                },
            )

        import numpy as np 
        out = {}
        for k, v in d.items():
            out[k] = v
            if k in name2sqr:
                out[k + ':std'] = np.sqrt(name2sqr[k] - v * v)
        for fmt in self.output_formats:
            #if self.comm.rank == 0:
            fmt.writekvs(out)

        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args):
        self._do_log(args)

    def warn(self, *args):
        self._do_log(("[WARNING]", *args))

    # Configuration
    # ----------------------------------------
    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

    def __del__(self):
        self.close()


def configure(
        dir: Union[str, None] = None,
        format_strs: Union[str, None] = None,
        date=False,
        **kwargs,
):
    if Logger.CURRENT is not None:
        Logger.CURRENT.close()

    if dir is None:
        if os.getenv("LOGGER_DIR"):
            dir = os.environ["LOGGER_DIR"]
        else:
            dir = osp.join(
                tempfile.gettempdir(),
                datetime.datetime.now().strftime("logger-%Y-%m-%d-%H-%M-%S-%f"),
            )
            print(f"using {dir} for recording.. set LOGGER_DIR to change the path")
            date = False

    if date:
        dir = os.path.join(dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    os.makedirs(dir, exist_ok=True)

    #import git
    #repo = git.Repo(search_parent_directories=True)
    #sha = repo.head.object.hexsha

    # choose log suffix based on world rank because otherwise the files will collide
    # if we split the world comm into different comms
    from tools import dist_utils as comm
    if comm.get_rank() == 0:
        log_suffix = ""
    else:
        # log_suffix = "-rank%03i" % comm.get_rank()
        log_suffix = None

    if isinstance(format_strs, str):
        format_strs = format_strs.split('+')
    format_strs = format_strs or default_format_strs(0)#comm.rank)

    if log_suffix is not None:
        output_formats = [make_output_format(f, dir, log_suffix, **kwargs) for f in format_strs]
        Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
        Logger.CURRENT.use_wandb = None
        for i in output_formats:
            if isinstance(i, WandbOutputFormat):
                Logger.CURRENT.use_wandb = i
        Logger.log_suffix = log_suffix
    else:
        output_formats = []

    #log("logger: logging to %s" % dir, 'Git sha @', sha)
    return dir


def is_configured():
    return Logger.CURRENT is not None


def default_format_strs(rank):
    if rank == 0:
        return ["stdout", "log", "csv"]
    else:
        return []


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger


# ================================================================


def _demo():
    configure()
    log("hi")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    log("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    log("^^^ should see b = -33.3")

    logkv("b", -2.5)
    dumpkvs()


# ================================================================
# Readers
# ================================================================


def read_json(fname):
    import pandas

    ds = []
    with open(fname, "rt") as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)


def read_csv(fname):
    import pandas

    return pandas.read_csv(fname, index_col=None, comment="#")


def read_tb(path):
    """
    path : a tensorboard file OR a directory, where we will find all TB files
           of the form events.*
    """
    import pandas
    import numpy as np
    from glob import glob
    import tensorflow as tf

    if osp.isdir(path):
        fnames = glob(osp.join(path, "events.*"))
    elif osp.basename(path).startswith("events."):
        fnames = [path]
    else:
        raise NotImplementedError(
            "Expected tensorboard file or directory containing them. Got %s" % path
        )
    tag2pairs = defaultdict(list)
    maxstep = 0
    for fname in fnames:
        for summary in tf.train.summary_iterator(fname):
            if summary.step > 0:
                for v in summary.summary.value:
                    pair = (summary.step, v.simple_value)
                    tag2pairs[v.tag].append(pair)
                maxstep = max(summary.step, maxstep)
    data = np.empty((maxstep, len(tag2pairs)))
    data[:] = np.nan
    tags = sorted(tag2pairs.keys())
    for (colidx, tag) in enumerate(tags):
        pairs = tag2pairs[tag]
        for (step, value) in pairs:
            data[step - 1, colidx] = value
    return pandas.DataFrame(data, columns=tags)


if __name__ == "__main__":
    _demo()
