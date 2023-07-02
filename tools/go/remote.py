import os
import argparse
import yaml
from yaml import Loader
import tempfile
PATH = os.path.dirname(os.path.abspath(__file__))

def go_remote(programs, args):
    template_path = os.path.join(PATH, '../bin', 'job.yml')
    if args.template:
        template_path = args.template

    if args.mpi_size is not None:
        args.gpu = int(args.mpi_size)
        args.cpu = args.cpu * int(args.mpi_size)

    config = yaml.load(open(template_path, 'r'), Loader)
    # config['metadata']['labels']['user'] = args.username
    container = config['spec']['template']['spec']['containers'][0]
    resources = container['resources']
    resources['limits']['cpu'] = args.cpu
    if args.memory is not None:
        resources['limits']['memory'] = f"{args.memory * args.multiple}Gi"
    resources['limits']['nvidia.com/gpu'] = args.gpu

    if args.memory_upper is None:
        args.memory_upper = args.memory
    resources['requests']['cpu'] = args.cpu
    if args.memory is not None:
        resources['requests']['memory'] = f"{args.memory * args.multiple}Gi"
    resources['requests']['nvidia.com/gpu'] = args.gpu

    arguments = container['args'][-1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', required=True)
    names = []
    import copy
    outs = {}
    for i in programs:
        i_args, remain = parser.parse_known_args(i.split(' '))
        name = f"{args.username}-job-{i_args.job_name}"
        config['metadata']['name'] = name
        names.append(name)
        i = ' '.join(remain)

        if args.multiple > 1:
            i = i.replace("python3", "go")
            opts = i.split(' ')
            for j in range(len(opts)):
                if opts[j] == '--path':
                    opts[j+1] = f"@'[\"{opts[j+1]}\"+\"_\"+str(i) for i in range({args.multiple})]'"
            opts = opts[:3] + ["--go", f"--n_proc {args.multiple}", "-o"] + opts[3:]
            i = ' '.join(opts)

        i = i.replace("python3", "python3")
        if args.mpi_size is not None:
            n_devices = int(args.mpi_size)
            devices  = ','.join(map(str,range(args.mpi_size)))
            i = f'CUDA_VISIBLE_DEVICES={devices} MPI_SIZE={n_devices} {i}'

        if args.workspace is None:
            args.workspace = os.path.join('/root', os.path.relpath(os.getcwd(), os.path.join(PATH, '../../../')))

        container['args'][-1] = arguments.replace("WORKSPACE", args.workspace).replace("CMD", i)

        print(container['args'][-1])
        print(yaml.dump(config))

        outs[name] = copy.deepcopy(config)

    assert len(names) == len(set(names)), f"exp_name must be unique!, {set(names)} for {len(names)} exps"

    paths = {}
    import datetime
    save_path = os.path.join(
        tempfile.gettempdir(),
        datetime.datetime.now().strftime(f"{args.save_path}-%Y-%m-%d-%H-%M-%S-%f"),
    )
    print('saving configs in', save_path)
    os.makedirs(save_path, exist_ok=True)
    for i in outs:
        paths[i] = os.path.join(save_path, i+'.yml')
        yaml.dump(outs[i], open(paths[i],'w'))

    if args.go:
        os.system("git commit -am 'update'")
        os.system("git push")

        if not args.silent:
            print("(use --silent to skip this prompt) Are you sure you want to run it? Please enter: yes to execute")
            com = input()
        else:
            com = 'yes'
        if com == 'yes':
            for i in paths:
                print(paths[i])
                os.system(f"kubectl create -f {paths[i]}")