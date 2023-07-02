# https://pytorch.org/tutorials/intermediate/dist_tuto.html
import os
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, fn, device, seed=True, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    print(f'{rank} CUDA initialize', os.environ['CUDA_VISIBLE_DEVICES'])
    dist.init_process_group(backend, rank=rank, world_size=int(os.environ['MPI_SIZE']))

    if seed:
        torch.manual_seed(rank)
        np.random.seed(rank)
        random.seed(rank)
    fn()

def is_distributed_training():
    return 'MPI_SIZE' in os.environ

def get_rank():
    if is_distributed_training():
        return dist.get_rank()
    return 0

def get_world_size():
    if is_distributed_training():
        return dist.get_world_size()
    return 1


def rank_print(*args, **kwargs):
    #if get_rank() == 0:
    #    print(*args, **kwargs)
    print('RANK' + str(get_rank()) + ':', *args, **kwargs)

def launch(main, seed=True, defualt_MPI_SIZE=None, default_device=None):

    if defualt_MPI_SIZE is not None and not is_distributed_training():
        os.environ['MPI_SIZE'] = str(defualt_MPI_SIZE)

    if default_device is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(default_device) 

    if is_distributed_training():
        mp.set_start_method("spawn")

        size = int(os.environ['MPI_SIZE'])
        devices = list(range(torch.cuda.device_count())) # by default, using all gpus
        if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ['CUDA_VISIBLE_DEVICES'])>0:
            # unless we manually specify the CUDA_VISIBLE_DEVICES
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

        processes = []
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, main, devices[rank % len(devices)], seed))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        main()


# sync the networks
def sync_networks(network):
    if is_distributed_training():
        params = network.parameters() if isinstance(network, torch.nn.Module) else [network]
        for param in params:
            dist.broadcast(param.data, 0)

# sync the networks
def sync_grads(network):
    if is_distributed_training():
        size = float(dist.get_world_size())
        params = network.parameters() if isinstance(network, torch.nn.Module) else [network]
        for param in params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


def weighted_mean(local_name2valcount):
    local_name2valcount = {
        name: (float(val), count)
        for (name, (val, count)) in local_name2valcount.items()
    }
    if is_distributed_training():
        all_name2valcount = [None for _ in range(get_world_size())]
        dist.all_gather_object(all_name2valcount, local_name2valcount)
    else:
        all_name2valcount = [local_name2valcount]

    if get_rank() == 0:
        from collections import defaultdict
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                name2sum[name] += val * count
                name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}