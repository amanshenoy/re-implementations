from utils import * 

# Masked Convolution where -
# Mask 'input' looks only in the past, mask 'middle' looks at the present + the past
class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type = 'middle', class_conditional = False, num_labels = 10,  **kwargs):
        assert mask_type == 'input' or mask_type == 'middle' 
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.populate_mask(mask_type)
        self.class_conditional = class_conditional
        
        if self.class_conditional:
            self.num_labels = num_labels
            self.class_cond_bias = nn.Linear(self.num_labels, self.out_channels) 

    def forward(self, input, oh_labels):
        batch_size = input.shape[0]
        out = f.conv2d(input, self.weight * self.mask, self.bias, self.stride, 
                            self.padding, self.dilation, self.groups)
        if self.class_conditional: 
            out = out + self.class_cond_bias(oh_labels).view(batch_size, -1, 1, 1)
        return out    
    
    def populate_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'middle': self.mask[:, :, k // 2, k // 2] = 1 

# Residual Block to repeat
class ResBlock(nn.Module):
    def __init__(self, in_channels, cc, **kwargs):
        super().__init__()
        self.block = nn.ModuleList([
            MaskedConv2d(in_channels = in_channels, out_channels = in_channels // 2, kernel_size = 1, class_conditional = cc, **kwargs), 
            nn.ReLU(),
            MaskedConv2d(in_channels = in_channels // 2, out_channels = in_channels // 2, kernel_size = 7, padding = 3, class_conditional = cc, **kwargs),
            nn.ReLU(),
            MaskedConv2d(in_channels = in_channels // 2, out_channels = in_channels, kernel_size = 1, class_conditional = cc, **kwargs)
        ])

    def forward(self, x, oh_labels):
        output = x
        for layer in self.block:
            if isinstance(layer, MaskedConv2d):
                output = layer(output, oh_labels)
            else: output = layer(output)
        return output + x

class ChannelLayerNorm(nn.LayerNorm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x = super().forward(x)
    return x.permute(0, 3, 1, 2).contiguous()

# PixelCNN implementation, cc is flag for class conditional - adding an extra class input in the forward
class PixelCNN(nn.Module):
    def __init__(self, cc, **kwargs):
        super().__init__()
        self.layers = [
            MaskedConv2d(mask_type = 'input', in_channels = 1, out_channels = 32, kernel_size = 7, padding = 3, class_conditional=cc, **kwargs),
            ChannelLayerNorm(32),
            nn.ReLU()
        ] + [
            ResBlock(in_channels = 32, cc = cc),
            ChannelLayerNorm(32),
            nn.ReLU()
        ] * 3 + [
            MaskedConv2d(in_channels = 32, out_channels = 2, kernel_size = 1, class_conditional=cc, **kwargs),
            nn.ReLU()
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, input, oh_labels):
        out = input
        for layer in self.model:
            if isinstance(layer, MaskedConv2d) or isinstance(layer, ResBlock):
                out = layer(out, oh_labels)
            else:
                out = layer(out)
        return out
    
    def loss(self, binary_input, oh_labels):
        return f.cross_entropy(self(binary_input, oh_labels), binary_input.squeeze().long())

    def train_params(self, num_epochs, loader, optimizer):
        n = 0
        for epoch in tqdm(range(num_epochs), desc = 'Training Progress'):
            for batch, labels in tqdm(loader, desc = "Epoch {} progress".format(epoch + 1)):
                bs = batch.shape[0]
                batch = binarize(batch, 0.5)
                optimizer.zero_grad()
                batch_in = batch.to(device).type(torch.float32)
                one_hot_batch = torch.zeros(bs, 10)
                one_hot_batch = torch.scatter(one_hot_batch, 1, labels[:, None], 1).to(device)
                
                loss = model.loss(batch_in, one_hot_batch)
                loss.backward()
                optimizer.step()
                writer.add_scalar("Training Loss (Nats per dim)", loss, n)
                n += 1     
        writer.close()

    def sample(self, num_samples = 5, image_size = 28, cond_num = 0):
        self.eval()
        samples = []
        n = 0
        cond_label = torch.zeros(1, 10).to(device)
        cond_label[0][cond_num] = 1

        for num_sample in tqdm(range(num_samples), desc = 'Sampling images'):
            empty_image = torch.zeros(1, 1, image_size, image_size).to(device)
            ar_vars = np.prod(empty_image.shape)
            running_image = empty_image
            
            for ar_var in tqdm(range(ar_vars), desc = 'Sampling autoregressive vars for image {}'.format(num_sample + 1)):
                logits_out = self(running_image, cond_label).permute(0, 2, 3, 1)
                x, y = ar_var // image_size, ar_var % image_size
                probs = torch.softmax(logits_out[0][x][y], dim = 0)
                dist = torch.distributions.categorical.Categorical(probs)
                sample_xy = dist.sample()
                running_image[0][0][x][y] = sample_xy

            samples.append(running_image[0])                       
        
        out_image = torch.cat(samples, dim = -1)
        writer.add_image("Sampled array for class {}".format(cond_num), out_image, 0)
        writer.close()
        samples = []
        return samples 

if __name__ == '__main__':
    args = parser()
    log_dir = os.path.join(str(args.logdir), str(args.run_num)) 
    writer = SummaryWriter(os.path.join(args.logdir, str(args.run_num)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} for {}".format(device, 'training' if args.type else 'sampling'))

    model = PixelCNN(cc = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    mnist = torchvision.datasets.MNIST('./data', download = args.type, train = True, transform = torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle = True) 
    model_path = os.path.join(args.save_dir, 'pixel_cnn.pth')
    num_epochs = args.epochs
    model.to(device)

    if args.type:
        model.train()
        model.train_params(num_epochs, loader, optimizer)
        torch.save(model.state_dict(), model_path)
        print("Trained model saved at {}".format(model_path))

    else:
        model.load_state_dict(torch.load(model_path))   
        for i in range(10):
            model.sample(num_samples = 5, cond_num=i)