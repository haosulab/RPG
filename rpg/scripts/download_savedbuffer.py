import os

pod_name = 'hza-job-denseantabl-6-seed-3-9g5ng'

for i in ['gaussian_seed1', 'gaussian_seed2', 'gaussian_seed3', 'mbsac_seed1', 'mbsac_seed2', 'mbsac_seed3']:
    os.system(f"kubectl cp {pod_name}:/cephfs/hza/buffers/{i} data/savedbuffer/{i}")
