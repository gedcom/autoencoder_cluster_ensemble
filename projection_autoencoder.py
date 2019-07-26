import csv
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import decomposition
from sklearn import preprocessing
import argparse
from os.path import join
parser = argparse.ArgumentParser(description='Autoencoder')

parser.add_argument('--path',
                    help='Path of input file',
		    default=None)
parser.add_argument('--save_dir',
                    help='Path of output directory', 
                    default=None)
parser.add_argument('--random_proj_time',
                    type=int,
                    help='Number of random projection samples to generate', 
                    default=2)
parser.add_argument('--random_proj_dim',
                    type=int,
                    help='Number of genes to be selected for randomly projected samples', 
                    default=None)
parser.add_argument('--ae_dim',
                    type=int,
                    help='Size/dimension of encoded feature space', 
                    default=16)
parser.add_argument('--hidden_dims',
                    type=int,
                    nargs='+',
                    help='Widths of hidden layers in autoencoder', 
                    default=[128])
parser.add_argument('--learning_rate',
                    type=float,
                    help='Learning rate for backpropagation',
                    default=0.001)


class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, reduced_dim):
        super(Autoencoder, self).__init__()
        len_hidden = len(hidden_dims)
        structure_encoder = []
        for i in range(len_hidden):
            bef_dim = hidden_dims[i-1] if i>0 else input_dim
            nxt_dim = hidden_dims[i]
            structure_encoder.append(nn.Linear(bef_dim, nxt_dim))
            structure_encoder.append(nn.LeakyReLU())
        structure_encoder.append(nn.Linear(hidden_dims[-1], reduced_dim))
        self.encoder = nn.Sequential(*structure_encoder)
        structure_decoder = []
        for i in range(len_hidden):
            bef_dim = hidden_dims[-i] if i>0 else reduced_dim
            nxt_dim = hidden_dims[-i-1]
            structure_decoder.append(nn.Linear(bef_dim, nxt_dim))
            structure_decoder.append(nn.LeakyReLU())
        structure_decoder.append(nn.Linear(hidden_dims[0], input_dim))
        self.encoder = nn.Sequential(*structure_encoder)
        self.decoder = nn.Sequential(*structure_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
            
def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)

def cal_pca_score(X, dim):
    criterion = nn.MSELoss()
    pca = decomposition.PCA(n_components=dim, copy=True, whiten=False)
    pca.fit(X)
    reconstructed = pca.inverse_transform(pca.transform(X))
    loss = criterion(to_tensor(reconstructed), to_tensor(X))
    print('pca loss is {}'.format(loss))
    
def solve(data, random_proj_dim, hidden_dims, ae_dim, learning_rate):
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    
    cal_pca_score(data, ae_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data = to_tensor(data).to(device)
    
    num_epochs = 100
    batch_size = 32
    print('num_epochs = {}'.format(num_epochs))
    print('learning_rate = {}'.format(learning_rate))
    print('batch_size = {}'.format(batch_size))
    
    N, M = data.shape
    model = Autoencoder(input_dim=M, hidden_dims=hidden_dims, reduced_dim=ae_dim)
    model.apply(init_weights)
    print(model)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(num_epochs)):
        for cur_data in dataloader:
            #cur_data = to_tensor(cur_data).to(device)
            output = model(cur_data)
            loss = criterion(output, cur_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print('epoch [{}/{}], loss:{:.4f}'
        #  .format(epoch + 1, num_epochs, loss.data[0]))
    
    with torch.no_grad():
       # reduced_features = model.encoder(to_tensor(data).to(device))
        reduced_features = model.encoder(data)
        reconstructed = model.decoder(reduced_features)
       # loss = criterion(to_tensor(data).to(device), reconstructed)
        loss = criterion(data, reconstructed)

    return loss, reduced_features

def gen_data(all_feature_label, all_data, random_proj_dim):
    N, M = all_data.shape
    mask = [True]*random_proj_dim + [False]*(M - random_proj_dim)
    np.random.shuffle(mask)
    return all_feature_label[mask], np.array([x[mask] for x in all_data])

def ae(all_feature_label, all_data, random_proj_time, random_proj_dim, ae_dim, hidden_dims, learning_rate, save_dir):
    """Calculate reduced features
    Args:
        all_feature_label (list(str)): All feature labels
        all_data (a numpy array (N * M)):
                            Data which have changed to numpy format
                            N is the number of cells,
                            M is the number of features
        random_proj_time (int): Number of random projection samples to generate
        random_proj_dim (int): Number of genes to be selected for randomly projected samples
        ae_dim (int): Size/dimension of encoded feature space
        hidden_dims ([int]): Dimensions of hidden layers
    Returns: 
        reduced_features (a list of numpy arrays [(N * ae_dim)]): 
                            containing the encoded values for each random projection
        losses ([float]): Reconstruction error for each random projection/autoencoder
        feature_labels ([str]): The label generated by random_project
    """
    feature_labels = []
    reduced_features = []
    losses = []
    for it in range(random_proj_time):
        feature_label, data = gen_data(all_feature_label, all_data, random_proj_dim)
        print('data shape={}'.format(data.shape))
        feature_labels.append(feature_label)
        loss, reduced_feature = solve(data, random_proj_dim, hidden_dims, ae_dim, learning_rate)
        reduced_features.append(reduced_feature)
        if save_dir is not None:
            #np.save(join(save_dir, str(it)+'.npy'),
            #        reduced_feature)
            np.savetxt(join(save_dir, str(it)+'.csv'), reduced_feature, delimiter = ",")
        losses.append(loss)
    
    return reduced_features, losses, feature_labels
        
def readcsv(path):
    label = []
    data = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for raw in reader:
            label.append(raw[0])
            data.append([float(x) for x in raw[1:]])
    label = np.array(label)
    data = np.array(data).T
    return label, data

def test(args):
    print(args)
    path = args.path
    random_proj_time = args.random_proj_time
    random_proj_dim = args.random_proj_dim
    ae_dim = args.ae_dim
    hidden_dims = args.hidden_dims#[256,128]
    learning_rate = args.learning_rate

    all_feature_label, all_data = readcsv(path)

    if random_proj_dim is None:
        random_proj_dim = min(int(all_data.shape[1] * 0.8))
    elif random_proj_dim > all_data.shape[1]:
        random_proj_dim = all_data.shape[1]

    print('dataset with shape {}'.format(all_data.shape))
    reduced_features, losses, feature_labels = ae(all_feature_label, all_data, 
                              random_proj_time=random_proj_time, 
                              random_proj_dim=random_proj_dim, 
                              ae_dim=ae_dim,
                              hidden_dims=hidden_dims,
                              learning_rate=learning_rate,
                              save_dir=args.save_dir)
    #print('reduced_features = {}'.format(reduced_features))
    print('losses = {}'.format(losses))
    return reduced_features, losses, feature_labels

if __name__== '__main__':
    args = parser.parse_args()
    test(args)
