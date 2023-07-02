import sys
import termcolor
import types
import inspect


def exception_with_traceback(frame):
    # https://martinheinz.dev/blog/66
    errors = '' 
    while frame is not None:
        source, firstline = inspect.findsource(frame.f_code)
        if frame.f_lineno == len(source):
            break
        code = source[frame.f_lineno-1].strip()
        
        errors += f"\n  File {frame.f_code.co_filename}, line {frame.f_lineno}, line {frame.f_code.co_name} \n       {code}"
        try:
            frame = frame.f_back
        except ValueError as exc:
            break
    return errors
    #return exception_cls(message + termcolor.colored(errors, 'yellow'))


def frame_assert(cond, msg, trace, errorType=ValueError):
    if not cond:
        #source, firstline = inspect.findsource(frame.f_code)
        #code = source[frame.f_lineno].strip()
        #error_message = f"\n    {code} at {frame}."
        #raise ValueError(msg + f" in {self._name} from frame {error_message}")
        #raise exception_with_traceback(frame, errorType, msg)
        raise errorType(msg + termcolor.colored(trace(), 'yellow'))

def tensor2error(x):
    import torch
    if isinstance(x, torch.Tensor):
        return str(x)[:10] + '...' + ') of shape ' + str(x.shape) + ' of type ' + str(x.dtype)
    return str(x)