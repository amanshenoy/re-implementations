from utils import *

# Masked Linear Layer with populable masks
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
    
    def fill_mask(self, mask):
        self.mask.data.copy_(mask.T)
        assert self.mask.shape == self.weight.shape
    
    def forward(self, input):
        return f.linear(input, self.mask * self.weight, self.bias)

# MADE Model implementation 
class MADE(nn.Module):  
    def __init__(self, d, in_dim, one_hot_input = True, ordering = False, layers = [512, 512, 512]):
        super().__init__()
        self.d = d 
        self.hidden_sizes = layers
        self.in_dim = in_dim
        self.in_vars = np.prod(self.in_dim)
        self.ordering = ordering if ordering else list(range(self.in_vars))
        self.out_vars = self.in_vars * d
        self.nn_sizes = [self.in_vars * self.d if one_hot_input else self.in_vars] + layers + [self.out_vars]
        self.nn_list = []

        for in_shapes, out_shapes in zip(self.nn_sizes, self.nn_sizes[1:]):
            block = [MaskedLinear(in_shapes, out_shapes, True), nn.ReLU()]
            self.nn_list.extend(block)
        
        self.nn_list.pop()    
        self.model = nn.Sequential(*self.nn_list)
        self.one_hot_input = one_hot_input 
        self.m = {}
        self.seed = 10

    def populate_masks(self):
        L = len(self.hidden_sizes)
        rng = np.random.RandomState(self.seed)
        
        temp = torch.arange(self.in_vars) 
        self.m[-1] = torch.empty(self.in_vars * self.d)
        for num, element in enumerate(temp):
            self.m[-1][num * self.d: (num + 1) * self.d] = element

        for l in range(L):
            self.m[l] = torch.tensor(rng.randint(self.m[l-1].min(), self.in_vars-1, size=self.hidden_sizes[l]))
        
        self.masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        self.masks.append(self.m[L-1][:,None] < self.m[-1][None,:])

        if self.out_vars > self.in_vars:
            rel = self.in_vars * self.d if self.one_hot_input else self.in_vars 
            k = int(self.out_vars / rel)
            self.masks[-1] = torch.tensor(np.concatenate([self.masks[-1]]*k, axis=1))

        to_mask = [l for l in self.model.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(to_mask, self.masks):
            l.fill_mask(m)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        start_oh = to_one_hot(x[:, 0], self.d, device)
        
        for ar_var in range(x.shape[1] - 1):
            start_oh = torch.cat((start_oh, to_one_hot(x[:, ar_var], self.d, device)), dim = 1)
        
        nn_out = self.model(start_oh).view(bs, self.in_vars, self.d)
        nn_out = nn_out.permute(0, 2, 1).view(bs, self.d, *self.in_dim)
        return nn_out

    def loss(self, x):
        loss = f.cross_entropy(self(x), x.squeeze().long())
        return loss

    def train_params(self, num_epochs, loader, optimizer):
        self.populate_masks() 
        n = 0
        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            for point, label in tqdm(loader, desc='Epoch {} Progress'.format(epoch + 1)): 
                optimizer.zero_grad()
                loss = self.loss(point.to(device))
                loss.backward()
                optimizer.step()
                writer.add_scalar("Training Loss (Nats per dim)", loss, n)
                n += 1
        writer.close()

    def sample(self, num_iters = 10, im_size = 28):
        for num_iter in tqdm(range(num_iters), desc = "Sampling Images"):
            running_image = torch.zeros(1, 28, 28).to(device)
            probs = torch.softmax(self(seed_point).permute(0, 2, 3, 1), dim = -1)
            probs_x1 = probs[0][0][0]
            dist_x1 = torch.distributions.categorical.Categorical(probs_x1)
            img = 0
            running_dist = dist_x1

            for ar_var in tqdm(range(np.prod(running_image.shape)), desc = 'Sampling autoregressive vars for image {}'.format(num_iter + 1)):
                var = int(running_dist.sample())
                img.append(var)
                x, y = ar_var // im_size, ar_var % im_size
                running_image[0][x][y] = var
                probs = torch.softmax(self(running_image).permute(0, 2, 3, 1), dim = -1)
                probs_xy = probs[0][x][y]
                dist_xy = torch.distributions.categorical.Categorical(probs_xy)
                running_dist = dist_xy
            
            writer.add_image("Sampled image {}".format(num_iter), running_image, 0)
        writer.close()

if __name__ == '__main__':
    args = parser()
    log_dir = os.path.join(str(args.logdir), str(args.run_num)) 
    writer = SummaryWriter(os.path.join(args.logdir, str(args.run_num)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} for {}".format(device, 'training' if args.type else 'sampling'))

    model = MADE(2, (28, 28), layers = [512, 512])
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    mnist = torchvision.datasets.MNIST('./data', download = args.type, train = True, transform = torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle = True) 
    model_path = os.path.join(args.save_dir, 'made.pth')
    num_epochs = args.epochs
    model.to(device)

    if args.type:
        model.train_params(num_epochs, loader, optimizer)
        torch.save(model.state_dict(), model_path)
        print("Trained model saved at {}".format(model_path))

    else:
        model.load_state_dict(torch.load(model_path))   
        model.sample(args.num_samples)