from utils import *

# Any RNN followed by a linear layer, taking in pixel value + location information 
class AutoRegressiveRNN(nn.Module):
    def __init__(self, position_info = False, **kwargs):
        super().__init__()
        self.rnn = nn.GRU(**kwargs)
        self.linear = nn.Linear(kwargs['hidden_size'], 2)
        self.positional_info = position_info

    def forward(self, input):
        dist_in = torch.stack((input, 1 - input), dim = -1)
        
        if self.positional_info:  
            dist_in = torch.cat((dist_in, get_locations_to_append(input.shape[0], 784, 28).to(device)), dim = -1)
        
        rnn_out = self.rnn(dist_in)
        dist_out = self.linear(rnn_out[0])
        return dist_out

    def loss(self, input):
        targets = torch.cat((input.long()[:, 1:], torch.zeros(input.shape[0])[:, None].to(device)), dim = 1)
        return f.cross_entropy(self(input).permute(0, 2, 1), targets.long())

    def train_params(self, num_epochs, loader, optimizer):
        n = 0
        for epoch in tqdm(range(num_epochs), desc = 'Training Progress'):
            for batch, labels in tqdm(loader, desc = "Epoch {} progress".format(epoch + 1)):
                bs = batch.shape[0]
                batch = binarize(batch, 0.5)
                optimizer.zero_grad()
                batch_in = batch.to(device).type(torch.float32).view(bs, -1)
                
                loss = self.loss(batch_in)
                loss.backward()
                optimizer.step()
                writer.add_scalar("Training Loss (Nats per dim)", loss, n)
                n += 1
        writer.close()

    def sample(self, num_samples = 10, per_row = 5):
        count = 1
        self.eval()
        for num_sample in tqdm(range(num_samples + 1), desc="Sampling {} images".format(num_samples)):
            running_image = torch.zeros(1, 784).to(device)

            for ar_var in tqdm(range(784 - 1), desc="Sampling Auto-regressive variables for image {}".format(num_sample + 1)):
                logits_out = self(running_image)
                probs = torch.softmax(logits_out[0][ar_var], dim = 0)
                dist = torch.distributions.categorical.Categorical(probs)
                sampled = dist.sample()
                running_image[0][ar_var + 1] = sampled
            
            if (num_sample) % per_row == 0:
                running_array = running_image.view(1, 28, 28)
            else:
                running_array = torch.cat((running_array, running_image.view(1, 28, 28)), dim = 2)
            
            if (num_sample + 1) % per_row == 0:
                writer.add_image("Image array {}".format(count), running_array, count)
                count += 1
        writer.close()
        
if __name__ == '__main__':
    args = parser()
    log_dir = os.path.join(str(args.logdir), str(args.run_num)) 
    writer = SummaryWriter(os.path.join(args.logdir, str(args.run_num)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} for {}".format(device, 'training' if args.type else 'sampling'))

    model = AutoRegressiveRNN(position_info = True, input_size = 4, hidden_size = 256, num_layers = 1, batch_first=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    mnist = torchvision.datasets.MNIST('./data', download = args.type, train = True, transform = torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle = True) 
    model_path = os.path.join(args.save_dir, 'ar_rnn.pth')
    num_epochs = args.epochs
    model.to(device)

    if args.type:
        model.train_params(num_epochs, loader, optimizer)
        torch.save(model.state_dict(), model_path)
        print("Trained model saved at {}".format(model_path))

    else:
        model.load_state_dict(torch.load(model_path))   
        model.sample(args.num_samples)