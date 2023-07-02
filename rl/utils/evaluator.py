from operator import mod
import json
import tqdm
import cv2
import numpy as np
from tools import Configurable, merge_inputs
from tools.utils import animate, logger


class Evaluator(Configurable):
    keep_highest = True
    monitor_name = 'score'
    def __init__(self, data, cfg=None):
        super(Evaluator, self).__init__()
        self.dataset = data

    def __call__(self, model):
        raise NotImplementedError

    def dump(self, path, eval_status):
        raise NotImplementedError


class RLEvaluator(Evaluator):
    monitor_name = 'test-return'

    def __init__(
        self,
        envs,
        cfg=None,
        num_episodes=10,
        render_episodes=1,
    ):
        super(RLEvaluator, self).__init__(envs)
        self.envs = envs

    def __call__(self, model, **kwargs):
        cfg = merge_inputs(self._cfg, **kwargs)

        outs = {
            'return': [],
            'success': [],
            'steps': []
        }
        assert cfg.render_episodes <= cfg.num_episodes

        if cfg.render_episodes:
            images = []

        for episode_id in tqdm.trange(cfg.num_episodes):
            episode_steps = 0
            episode_returns = 0

            def render_image():
                image = self.envs.render(mode='rgb_array', id=0)[0]
                cv2.putText(image,
                    f'{episode_id}:{episode_steps} r{episode_returns}',
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                """
                # NOTE: this is too slow
                plt.imshow(image)
                plt.text(1, 1, f'{episode_id}:{episode_steps} r{episode_returns}')

                with io.BytesIO() as buff:
                    plt.savefig(buff, format='png')
                    buff.seek(0)
                    image = plt.imread(buff)
                    plt.clf()
                    """

                return image

            obs = self.envs.reset(0)
            if episode_id < cfg.render_episodes:
                images.append(render_image())


            while True:
                action = model.select_action(obs[0])
                obs, reward, done, info = self.envs.step([action], id=0)
                episode_returns += reward[0]
                episode_steps += 1

                if episode_id < cfg.render_episodes:
                    images.append(render_image())

                if done:
                    break

            if 'success' in info:
                outs['success'].append(info['success'])
            outs['return'].append(episode_returns)
            outs['steps'].append(episode_steps)

        output = {
            'test-' + key: np.mean(val)
            for key, val in outs.items() if len(val) > 0
        }

        logger.logkvs_mean(output)

        if cfg.render_episodes:
            output['video'] = images

        return output


    def dump(self, path, eval_status):
        eval_status = dict(eval_status) # shallow copy to allow pop
        if 'video' in eval_status:
            video = eval_status.pop('video')
            animate(video, path+'.mp4', _return=False)

        with open(path + '.json', 'w') as f:
            json.dump(eval_status, f, indent=2)