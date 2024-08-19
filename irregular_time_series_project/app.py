from ast import Num
from re import X
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import numpy as np 
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns 
import scipy
from scipy.interpolate import CubicSpline
import torch.utils.data as data
import torch.nn as nn


df = pd.read_csv("C:\\Users\\Batuhan\\Desktop\\irregular_time_series_project\\Price_Test_GP_Stock_DSE.csv")

#print(df)

df.columns = ['Date','Open','High','Low','Close','Volume','arranged_timestamp']
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

min_timestamp = df['Date'].min()
max_timestamp = df['Date'].max()

df['arranged_timestamp'] = (df['Date'] - min_timestamp) / (max_timestamp - min_timestamp)



output_file_path = 'C:\\Users\\Batuhan\\Desktop\\irregular_time_series_project\\Price_Test_GP_Stock_DSE.csv'
df.to_csv(output_file_path, index = False)

# print(f'Timestamp values scaled betwwen 0 and 1 and saved to {output_file_path}')
# print(df)

# plt.plot(df['arranged_timestamp'])
# plt.show()

feature = df['arranged_timestamp'].values
df.reset_index(drop=True, inplace=True)

# split data into train and test parts

train_size = int(len(df) * 0.50)
test_size = len(df) - train_size
train, test = df['arranged_timestamp'][:train_size].values , df['arranged_timestamp'][train_size:].values


def create_dataset(dataset, lookback):
    X, y  = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback].astype(np.float32)
        target = dataset[i+1:i+lookback+1].astype(np.float32)
        X.append(feature)
        y.append(target)
    return (torch.tensor(X), torch.tensor(y))    

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



class PredictingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        lstm_output, (last_hidden_state, _) = self.lstm(x)
        
        # Adjust the indexing based on the structure of lstm_output
        if len(lstm_output.shape) == 3:
            last_hidden_state = lstm_output[:, -1, :]
        elif len(lstm_output.shape) == 2:
            last_hidden_state = lstm_output
        else:
            raise ValueError("Unexpected LSTM output structure")
        
        x = self.linear(last_hidden_state)
        return x

        '''
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x
        '''

model = PredictingModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
        # shift train predictions for plotting
    train_plot = np.ones_like(df) * np.nan
    y_pred = model(X_train)

    
    if len(y_pred.shape) == 3:
        y_pred = y_pred[:, -1, :]
    elif len(y_pred.shape) == 2:
        y_pred = y_pred.squeeze()
    else:
        raise ValueError("Unexpected shape of y_pred")

     # y_pred = y_pred[:, -1, :]               

    if len(model(X_train).shape) == 3:
        train_plot[lookback:train_size] = model(X_train)[:, -1, :].squeeze().numpy()
    elif len(model(X_train).shape) == 2:
        train_plot[lookback:train_size, 1] = model(X_train).squeeze().numpy()
    else:
        raise ValueError("Unexpected shape of model(X_train)")
    
    # train_plot[lookback:train_size] = model(X_train)[:, -1, :]


    # shift test predictions for plotting
    test_plot = np.ones_like(df) * np.nan
    if len(model(X_test).shape) == 3:
        test_plot[train_size+lookback:len(df)] = model(X_test)[:, -1, :].squeeze().numpy()
    elif len(model(X_test).shape) == 2:
        test_plot[train_size+lookback:len(df), 1] = model(X_test).squeeze().numpy()
    else:
        raise ValueError("Unexpected shape of model(X_test)")

    # test_plot[train_size+lookback:len(df)] = model(X_test)[:, -1, :]

# plot
plt.plot(df['arranged_timestamp'])
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()




'''

def _solve_cde(x,y):
        

        coeffs = y.reshape(-1,1)

        t_interp = np.linspace(0, 1, 100)
        cs = CubicSpline(x, y)
        y_interp = cs(t_interp)

        

        #print(coeffs.shape)

        input_channels = 1
        hidden_channels = 4  
        output_channels = 10  



        class predictingModel(torch.nn.Module):
            def __init__(self):
                super(predictingModel, self).__init__()
                self.linear = torch.nn.Linear(hidden_channels, hidden_channels * input_channels)

            def forward(self, t, z):
                batch_dims = z.shape[:-1]
                return self.linear(z).tanh().view(*batch_dims, hidden_channels, input_channels)


        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.initial = torch.nn.Linear(input_channels, hidden_channels)
                self.func = predictingModel()
                self.readout = torch.nn.Linear(hidden_channels, output_channels)

            def forward(self, t, y):
                x = t
                t = torch.linspace(0, 1, coeffs.shape[0], dtype=torch.float32)

                X = torchcde.CubicSpline(x, y)
                t_interp = torch.linspace(0, 1, 100)  # Adjust the number of points as needed
                y_interp = X(t_interp)


                X0 = y_interp[0].unsqueeze(0).unsqueeze(-1)
                z0 = self.initial(X0)
                zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=t_interp)
                zT = zt[..., -1, :]  
                return self.readout(zT)

        model = Model()

        return model(torch.Tensor(x),torch.Tensor(y))

y = np.array(df['Close'])
x = np.arange(len(df['Date'])) 


_solve_cde(x,y)

'''



















'''

class F(CDEFunc):
    def __init__(self, input_dim, hidden_dim):
        super(F, self).__init__(input_dim, hidden_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, t, z):

        return self.net(z)
    
your_input_dim = 5
your_hidden_dim = 10

cde_block = cde_block(input_dim = your_input_dim, hidden_dim = your_hidden_dim)
model = NeuralCDE(cde_block)    

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.MSELoss()

inputs = torch.tensor(df[['Open', 'High', 'Low', 'Close', 'Volume']].values, dtype = torch.float32)
targets = torch.tensor(df['Volume'].values(), dtype=torch.float32)

dataloader = DataLoader(df, batch_size=64)

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()



model.eval()
with torch.no_grad():
    predictions = model(test_inputs)
'''





































# print(df)





'''                                           
     date_train, date_test, open_train, open_test, high_train, high_test, low_train, low_test, close_train, close_test, volume_train, volume_test = train_test_split(
    df['Date'],df['Open'],df['High'],df['Low'],df['Close'],df['Volume'],test_size=0.4, random_state=0)
# verinin %40 ı test kısmına %60 eğitim kısmına atandı.


model = LinearRegression().fit(date_train, open_train, high_train, low_train, close_train, volume_train)
model.score(x_train, y_train)
model.score(x_test, y_test)
    '''

# print(df.head())























