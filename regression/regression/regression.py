import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt

#建立數據
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
# x (tensor), shape=(100, 1)
#unsqueeze>>把一維的數據變成二維的
#[1,2,3,4]>>[[1,2,3,4]]

y = x.pow(2) + 0.2*torch.rand(x.size())                 
# noisy y (tensor), shape=(100, 1)

'''
#將數據畫出
plt.scatter(x.data.numpy(), y.data.numpy())
#scatter散點圖
plt.show()
'''

#架構
class Net(torch.nn.Module):  # 繼承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):  #初始化參數
        super(Net, self).__init__()     #繼承
        
        # 定義每層
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output)
        #n_feature>>n_hidden>>n_output

    def forward(self, x):   # 正向傳播
        x = F.relu(self.hidden(x))     #activation
        x = self.predict(x)             
        return x

#神經架構
net = Net(n_feature=1, n_hidden=10, n_output=1)


"""
print(net)  # net 的结构
>>>

Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""


# optimizer 訓練工具(優化器)
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  
# 傳入 net 的所有參數, 學習率

loss_func = torch.nn.MSELoss()      
# MSELoss>>>均方差(標準差)



plt.ion() 
plt.show()

for t in range(100): #訓練次數
    prediction = net(x)     # x:訓練數據

    loss = loss_func(prediction, y)     # 算loss   (prdiction在前)

    optimizer.zero_grad()   
    # 清空上一步的grad

    loss.backward()         
    # 反向傳播, 算出新的grad
    
    optimizer.step()        
    # 更新參數
    if t % 5 == 0:# plot and show learning process
         plt.cla()
         plt.scatter(x.data.numpy(), y.data.numpy())
         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
         plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
         plt.pause(0.1)

plt.ioff()
plt.show()
