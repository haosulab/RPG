import tqdm
from torchvision import datasets, transforms

dataset = datasets.MNIST('../data', train=True, transform=
                         transforms.Compose(
                            [transforms.Resize((32, 32)),
                             transforms.ToTensor()]
                         ),
                         download=True)

import torch
from torch import nn
from tools.nn_base import Network


class Encoder(Network):
    def __init__(self, latent_dim, cfg=None):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.linear = nn.Linear(64, latent_dim)

        #self.embedder = nn.Sequential(
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, context):
        x = self.act(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.act(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.act(self.conv3(x))
        x = x.mean(axis=-1).mean(axis=-1)
        return self.linear(x)


class Decoder(Network):
    def __init__(self, latent_dim, cfg=None):
        super().__init__()

        self.linear = nn.Linear(latent_dim, 64)

        self.conv1 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, 1)
        self.conv3 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1)
        self.out = nn.Conv2d(16, 1, 1, 1)

        #self.embedder = nn.Sequential(
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, context):
        x = self.act(self.linear(x).view(-1, 64, 1, 1))
        x = self.act(self.conv1(x))
        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x = self.act(self.conv2(x))
        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x = self.act(self.conv3(x))
        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x = self.act(self.conv4(x)[:, :, :28, :28])
        x = self.out(x)
        return x


# class Unet(Network):
#     def __init__(self, cfg=None):
#         super().__init__()
#         self.embed = SinusoidalPositionEmbeddings(16)
#         self.encoder = Encoder(128)
#         self.decoder = Decoder(128 + 16)
#         #self.conv0 = nn.Conv2d(1, 64, 3, 1, padding=1)

#         #self.conv1 = nn.Conv2d(64 + 128 + 16, 64, 3, 1, padding=1)
#         #self.conv2 = nn.Conv2d(64, 32, 3, 1, padding=1)
#         self.conv3 = nn.Conv2d(32, 1, 3, 1, padding=1)

#         #self.embedder = nn.Sequential(

#     def forward(self, x, context, t):
#         if len(t.shape) == 0:
#             t = torch.ones(x.shape[0], device=x.device) * t

#         #feature = self.encoder(x, None)[:, :, None, None].repeat(1, 1, 28, 28) # input a global feature ..
#         #embed = self.embed(t)[:, :, None, None].expand(-1, -1, 28, 28)
#         #x = torch.relu(self.conv1(torch.cat((self.conv0(x), feature, embed), 1)))
#         #x = torch.relu(self.conv2(x))
#         #x = self.conv3(x)
#         feature = self.encoder(x, context)
#         embed = self.embed(t)
#         f = torch.cat((feature, embed), 1)
#         return self.decoder(f, context)


from generative.diffusion_models import Denoiser

from torch import nn
import torch
from generative.vae import VAE, DiagGaussian, HiddenDDIM, DDIMScheduler


train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

# encoder = Encoder(256)
# decoder = Decoder(128)
# vae = VAE(encoder, decoder, DiagGaussian()).cuda()

input_fn = lambda x, y: (x-0.5)/0.5
output_fn = lambda x, y: x * 0.5 + 0.5
#unet = Unet()
ddim = DDIMScheduler()
unet = Denoiser(1, 1, len(ddim.timesteps), 32)
diffusion = HiddenDDIM(unet, ddim)

vae = VAE(input_fn, output_fn, diffusion).cuda()

from torch.optim import Adam, AdamW
optim = AdamW(vae.parameters(), lr=0.0001)


weights = {'kl': 0.1}

# import tqdm
# for i in train_loader:
#     j = i
#     break
for epoch in range(1000):
    print('epoch', epoch)
    losses_all = {}
    total = 0
    #for _ in tqdm.tqdm(range(100)):
    for i in tqdm.tqdm(train_loader, total=len(train_loader)):
        # i = j # fit single ..
        optim.zero_grad()
        i = i[0]
        i = i.cuda()
        out, losses = vae(i, None)
        loss = 0.
        for k, v in losses.items():
            losses_all[k] = losses_all.get(k, 0) + v.item()
            loss += weights.get(k, 1.) * v

        loss.backward()
        optim.step()

        total += 1


    import matplotlib.pyplot as plt
    inp = i[:10].reshape(-1, 32, 32).reshape(-1, 32)
    out = out[:10].reshape(-1, 32, 32).reshape(-1, 32)
    c = torch.cat([inp, out], axis=1)
    plt.imshow(c.detach().cpu().numpy().clip(0., 1.))
    plt.savefig('test.png')
    plt.clf()

    with torch.no_grad():
        s = vae.sample(100, None).view(10, 10, 32, 32).permute(0, 2, 1, 3).reshape(320, 320)
    print(s.min(), s.max())
    plt.imshow(s.detach().cpu().numpy().clip(0., 1.))
    plt.savefig('cur.png')
    plt.clf()

    print({k: v / total for k, v in losses_all.items()})