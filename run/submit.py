import os

ntls = "/home/litian/Desktop/llt/kubeutils/ntls.py"

date_today = "0612"
prefix = "2"
executer = os.system
# executer = print

for i in range(1, 2):
    for env in ["AdroitHammer"]: #  "MWStickPull", "AdroitHammer", ,"MWBasketBall"
        executer(
            f"python {ntls} --cmd='bash run.sh mbrpg.py {env} {prefix}{i}{date_today} rpgcv2' --prio regular"
        )

# for i in range(1, 2):
#     for env in ["AntPushDense", "CabinetDense"]: #  "MWStickPull", "AdroitHammer", ,"MWBasketBall"
#         executer(
#             f"python {ntls} --cmd='bash run.sh mbrpg.py {env} {prefix}{i}{date_today} dense' --prio regular"
#         )

print("ok")

# 03/01/2023 - fix routine.py, faster buffer
