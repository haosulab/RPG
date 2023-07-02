import os

pod_name = 'hza-try'

for i in ['cabinet']: #['block', 'cabinet', 'stickpull']:
    os.system(f"kubectl cp {pod_name}:/cephfs/hza/buffers/draw{i} data/savedeval/{i}")
