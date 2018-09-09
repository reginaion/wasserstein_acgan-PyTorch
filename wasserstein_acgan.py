
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torchsummary import summary
from tensorboardX import SummaryWriter 

import os
import random

root = '.\\'
taskname = '0000_exp0'


# In[2]:


cuda_available = torch.cuda.is_available()
random.seed(1)
torch.manual_seed(1)

print('random.seed:', random.randint(0,9))
print('torch.manual_seed:', torch.randint(low=0, high=9, size=[1]).long().item())
print('cuda_available:', cuda_available)


# In[3]:


writer = SummaryWriter(comment=taskname)


# In[4]:


lr_d = 2e-4
lr_g = 2e-4
betas = (0.5, 0.9)
lambda_penalty = 10
lambda_acgan = 1
one_hot = torch.eye(10).cuda() if cuda_available else torch.eye(10)


# In[5]:


latent_dim = 128
hidden_dim = 64
input_dim = 1
nclass = 10


# In[6]:


epochs = 10000
d_steps = 5
g_steps = 1
batch_size = 32
num_workers = 2
print_interval = 125
row_len = 16
download = False


# In[7]:


test_input = torch.rand(row_len*nclass, latent_dim)
test_label = one_hot[torch.cat([torch.ones(row_len)*j for j in range(nclass)]).long()]
if cuda_available:
    test_input = test_input.cuda()
    test_label = test_label.cuda()
print("test_input:", test_input)
print("test_label:", test_label)


# In[8]:


train = datasets.MNIST(root=os.path.join(root, 'data'),
                       train=True,
                       transform=transforms.ToTensor(),
                       download=download)
trainloader = DataLoader(train,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)


# In[9]:


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, nclass):
        super(Generator, self).__init__()
        self.preprocesses = nn.Sequential(nn.Linear(latent_dim + nclass, hidden_dim*4*4*4),
                                          nn.BatchNorm1d(hidden_dim*4*4*4),
                                          nn.ReLU(True))
        
        self.block1 = nn.Sequential(nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
                                    nn.BatchNorm2d(hidden_dim*2),
                                    nn.ReLU(True))
        
        self.block2 = nn.Sequential(nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.ReLU(True))
        
        self.block3 = nn.Sequential(nn.ConvTranspose2d(hidden_dim, 1, kernel_size=2, stride=2),
                                    nn.ReLU(True))
        
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x, l = x
        if not cuda_available:
            x = x.cpu()
            l = l.cpu()
        x = torch.cat([x, l], dim=1)
        x = self.preprocesses(x).view(-1, hidden_dim*4, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.tanh(x)


# In[10]:


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, nclass):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, kernel_size=2, stride=2),
                                    nn.LeakyReLU(),
                                    
                                    nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=2, stride=2),
                                    nn.LeakyReLU(),
                                    
                                    nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=1),
                                    nn.LeakyReLU())
        
        self.layer_decision = nn.Linear(hidden_dim*4*4*4, 1)
        self.layer_acgan = nn.Linear(hidden_dim*4*4*4, nclass)
    
    def forward(self, x):
        if not cuda_available:
            x = x.cpu()
        x = self.block1(x).view(-1, hidden_dim*4*4*4)
        return self.layer_decision(x).squeeze(), F.log_softmax(self.layer_acgan(x).squeeze(), dim=1)


# In[11]:


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


# In[12]:


def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if cuda_available else alpha
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    
    if cuda_available:
        interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    
    # modifity: ACGAN
    disc_interpolates, disc_labels = netD(interpolates)
    
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda_available else torch.ones(disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_penalty
    return gradient_penalty


# In[13]:


def get_generator_input():
    noise_vector = torch.rand(batch_size, latent_dim)
    one_hot_vector = torch.randint(low=0, high=9, size=[batch_size]).long()
    if cuda_available:
        noise_vector = noise_vector.cuda()
        one_hot_vector = one_hot_vector.cuda()
    return noise_vector, one_hot_vector


# In[14]:


G = Generator(latent_dim, hidden_dim, nclass).apply(weight_init).cuda() if cuda_available else Generator(latent_dim, hidden_dim, nclass).apply(weight_init)
D = Discriminator(input_dim, hidden_dim, nclass).apply(weight_init).cuda() if cuda_available else Discriminator(input_dim, hidden_dim, nclass).apply(weight_init)


# In[15]:


print('[1] Generator G:')
summary(G, input_size=[(latent_dim,),(nclass,)])
print('\n[2] Discriminator D:')
summary(D, input_size=(1, 28, 28))


# In[16]:


acgan_loss_func = nn.NLLLoss().cuda() if cuda_available else nn.NLLLoss()
d_optim = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
g_optim = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)


# In[17]:


check_label = lambda nj, j: torch.sum(torch.argmax(nj, dim=1) == j).item()/j.size(0)
mean_list = lambda l: sum(l)/len(l)
_str = '[Epoch: {:04d}/{:04d}], [Batch: {:04d}/{:04d}], [D_acc: {:.4f}], [Wasserstein_distance: {:.4f}]'


# In[18]:


d_label_loss_buffer = []
wasserstein_distance_buffer = []


# In[ ]:


for epoch in range(epochs):
    for _step, (i, j) in enumerate(trainloader):
        _iter = epoch*len(trainloader) + _step
        if cuda_available:
            i = i.cuda()
            j = j.cuda()
        # train discriminator
        D.zero_grad()
        
        # 1A
        d_real_data = i
        d_real_label = j
        
        d_real_decision, d_real_decision_label = D(d_real_data)
        d_real_decision = d_real_decision.mean()
        
        d_real_err = acgan_loss_func(d_real_decision_label, d_real_label)
        d_real_err = d_real_err.mean()
        
        # 1B
        d_noise_input, d_fake_label = get_generator_input()
        dg_fake_data = G([d_noise_input, one_hot[d_fake_label]]).detach()
        
        d_fake_decision, d_fake_decision_label = D(dg_fake_data)
        d_fake_decision = d_fake_decision.mean()
        
        d_fake_err = acgan_loss_func(d_fake_decision_label, d_fake_label)
        d_fake_err = d_fake_err.mean()
        
        # 1C
        gradient_penalty = calc_gradient_penalty(D, d_real_data.data, dg_fake_data.data)
        
        # 1D
        d_cost = d_fake_decision - d_real_decision + gradient_penalty
        d_err = ((d_real_err + d_fake_err)/2) * lambda_acgan
        (d_cost + d_err).backward()
        d_optim.step()
        
        # 1E
        wasserstein_distance = d_real_decision - d_fake_decision
        
        # 1F
        d_label_loss = (check_label(d_real_decision_label, d_real_label)+check_label(d_fake_decision_label, d_fake_label))/2
        
        if (_step+1)%d_steps == 0:
            for k in range(g_steps):
                # train generator
                G.zero_grad()
                
                # 2A
                g_noise_input, g_fake_label = get_generator_input()
                g_fake_data = G([g_noise_input, one_hot[g_fake_label]])
                
                gd_fake_decision, gd_fake_decision_label = D(g_fake_data)
                gd_fake_decision = gd_fake_decision.mean()
                
                gd_fake_err = acgan_loss_func(gd_fake_decision_label, g_fake_label)
                gd_fake_err = gd_fake_err.mean()
                
                # 2B
                g_cost = gd_fake_decision.mean() * -1
                g_err = gd_fake_err * lambda_acgan
                (g_cost + g_err).backward()
                g_optim.step()
                
                
                
            writer.add_scalar(os.path.join(taskname, 'wasserstein_distance'), wasserstein_distance, _iter)
            writer.add_scalars(os.path.join(taskname, 'cost'), {'D_cost': d_cost,
                                                                'G_cost': g_cost}, _iter)
            writer.add_scalars(os.path.join(taskname, 'err'), {'D_real_acc': check_label(d_real_decision_label, d_real_label),
                                                               'D_fake_acc': check_label(d_fake_decision_label, d_fake_label),
                                                               'G_err': check_label(gd_fake_decision_label, g_fake_label)}, _iter)
        
        
        
        d_label_loss_buffer.append(d_label_loss)
        wasserstein_distance_buffer.append(wasserstein_distance)
        
        if (_step+1)%print_interval == 0 or (_step+1)==len(trainloader):
            print(_str.format((epoch+1),
                              epochs,
                              (_step+1),
                              len(trainloader),
                              mean_list(d_label_loss_buffer if (_step+1)==len(trainloader) else d_label_loss_buffer[-print_interval:]),
                              mean_list(wasserstein_distance_buffer if (_step+1)==len(trainloader) else wasserstein_distance_buffer[-print_interval:])
                             ))
            if (_step+1)==len(trainloader):
                print('End')
                d_label_loss_buffer = []
                wasserstein_distance_buffer = []
    image = make_grid(G([test_input, test_label]).detach(), nrow=16)
    writer.add_image(os.path.join(taskname, 'Image'), image, epoch)


# In[20]:


writer.close()

