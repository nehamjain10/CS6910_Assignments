# %%
import torch
from torch import nn, optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import csv

class MLFFNN(nn.Module):
    """
    Class of Multi Layer Feed Forward Neural Network (MLFFNN)
    """
    def __init__(self, hidden_dim1=32,hidden_dim2=32) :
        super(MLFFNN, self).__init__()
        torch.manual_seed(3)
        # adding linear and non-linear hidden layers
        self.mlffnn = nn.Sequential(nn.Linear(INPUT_DIM, hidden_dim1),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim1, hidden_dim2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim2, NUM_CLASSES))
        
    def forward(self, X):
        y = self.mlffnn(X)
        return y


def plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,is_type="train",loss_or_accuracy="loss"):
    plt.figure(figsize=(10,8),dpi=300)
    print(epochs[0],len(loss_delta[is_type]))
    plt.plot(range(epochs[0]+1),loss_delta[is_type],label="Delta")
    plt.plot(range(epochs[1]+1),loss_ada_delta[is_type],label="Adaptive Delta")
    plt.plot(range(epochs[2]+1),loss_adam[is_type],label="Adam")
    plt.legend()
    if is_type=="train":
        plt.title(f"Training {loss_or_accuracy} vs Number of epochs with Learning Rate: {lr}")
        plt.savefig(f"figs/training_loss_{lr}.png")
    else:
        plt.title(f"Validation {loss_or_accuracy} vs Number of epochs with Learning Rate: {lr}")
        plt.savefig(f"figs/validation_{loss_or_accuracy}_{lr}.png")

def plot_confusion_matrix(lr,model_type,dataloader):
    model = torch.load(f'{model_type}')
    model.eval()
    with torch.no_grad():
        confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for X_test, y_test in dataloader:
            X_test = X_test.type(torch.float32).to(device)
            y_test = y_test.type(torch.long).to(device)
            score = model(X_test)
            score = score.squeeze(dim=1)
            y_test = y_test.squeeze(dim=1)
            max_indices = torch.argmax(score, dim=1)
            for t, p in zip(y_test.cpu(), max_indices.cpu()):
                confusion_matrix[t.long(), p.long()] += 1
        plt.figure(figsize=(10,8),dpi=300)
        sns.heatmap(confusion_matrix.numpy(), annot=True, xticklabels=idx_to_labels, yticklabels=idx_to_labels)  
        plt.title(f"Confusion Matrix with Learning Rate: {lr}")
        plt.savefig(f"figs/confusion_matrix_{lr}.png")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Code for loading the data
file_path_image_data = "task2b/single_label_image_dataset/image_data_dim60.txt"
file_path_image_labels = "task2b/single_label_image_dataset/image_data_labels.txt"

image_data = np.loadtxt(file_path_image_data)
labels = np.loadtxt(file_path_image_labels)

group_data = {1:0,2:1,3:2,5:3,6:4}
idx_to_labels = ["Forest","Highway","Inside","OpenCountry","Street"]

NUM_CLASSES = 5
MAX_EPOCHS = 100
BATCH_SIZE = 1

INPUT_DIM  = image_data.shape[1]
image_data = image_data[np.in1d(labels, list(group_data.keys()))]
labels = labels[np.in1d(labels, list(group_data.keys()))]

labels = np.vectorize(group_data.get)(labels)

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(image_data,labels, train_size=0.7,random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.67, random_state=42)

X_train, y_train, X_valid, X_test, y_valid, y_test = map(torch.tensor, [X_train, y_train, X_valid, X_test, y_valid, y_test])




# %%
accuracy_metrics = {
    'train': [],
    "val": []
}
loss_metrics = {
    'train': [],
    "val": []
}

# %%
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_valid.unsqueeze(1), y_valid.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_test.unsqueeze(1), y_test.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)



lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
criterion = nn.CrossEntropyLoss()

def train_model(optimizer,model,model_type="model_delta"):
    model.train()
    loss_metrics = {
    'train': [],
    "val": []
    }
    accuracy_metrics = {
    'train': [],
    "val": []
    }
    no_improvement=0
    best_loss = 100000
    for epoch in range(MAX_EPOCHS):
        train_loss = []
        correct_classified = 0
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.long).to(device)

            optimizer.zero_grad()

            score = model(X_train)
            score = score.squeeze(dim=1)
            y_train = y_train.squeeze(dim=1)
            
            loss = criterion(input=score, target=y_train)
            loss.backward()
            train_loss.append(loss.item())
            max_indices = torch.argmax(score, dim=1)
            correct_classified +=  (max_indices == y_train).float().sum()

            optimizer.step()

        accuracy_metrics["train"].append(correct_classified.item() / len(train_dataloader.dataset))
        # Validation
        validation_loss = []
        correct_classified = 0

        with torch.no_grad():
            for X_valid, y_valid in val_dataloader:
                model.eval()
                X_valid = X_valid.type(torch.float32).to(device)
                y_valid = y_valid.type(torch.long).to(device)
                
                score = model(X_valid)
                score = score.squeeze(dim=1)
                
                y_valid = y_valid.squeeze(dim=1)
                loss = criterion(input=score, target=y_valid)
                validation_loss.append(loss.item())

                max_indices = torch.argmax(score, dim=1)
                correct_classified +=  (max_indices == y_valid).float().sum()
        
        accuracy_metrics["val"].append(correct_classified.item() / len(val_dataloader.dataset))
        
        loss_metrics["train"].append(np.average(train_loss))
        loss_metrics["val"].append(np.average(validation_loss))
    
        if np.average(validation_loss) < best_loss:
            best_loss = np.average(validation_loss)
            torch.save(model, f'weights/{model_type}.pth')
            no_improvement=0
        else:
            no_improvement+=1
            if no_improvement==4:
                loss_metrics["train"] = loss_metrics["train"][:-no_improvement]
                break
    if no_improvement>0:  
        loss_metrics["train"] = loss_metrics["train"][:-no_improvement]
        loss_metrics["val"] = loss_metrics["val"][:-no_improvement]
        
        accuracy_metrics["train"] = accuracy_metrics["train"][:-no_improvement]
        accuracy_metrics["val"] = accuracy_metrics["val"][:-no_improvement]
        
    return loss_metrics, accuracy_metrics,epoch-no_improvement


    
hidden_dim_1 = [8,16,32,64]  
hidden_dim_2 = [8,16,32,64]
lrs = [1e-5,1e-4, 1e-3, 1e-2]

delta_loss = []
ada_delta_loss = []
adam_loss = []


f = open("validation_results.csv", "a")
csvwriter = csv.writer(f)

for hid_dim1 in hidden_dim_1:
    for hid_dim2 in hidden_dim_2:
        for lr in lrs:
            
            model_delta = MLFFNN(hid_dim1,hid_dim2).to(device)
            model_ada_delta = MLFFNN(hid_dim1,hid_dim2).to(device)
            model_adam = MLFFNN(hid_dim1,hid_dim2).to(device)
            
            optimizer_delta = optim.SGD(model_delta.parameters(), lr=lr, momentum=0)
            optimizer_ada_delta = optim.SGD(model_ada_delta.parameters(),lr=lr,momentum=.8)
            optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr)

            loss_delta,acc_delta,epoch_delta = train_model(optimizer_delta, model_delta,"model_delta")
            loss_ada_delta,acc_ada_delta,epoch_ada_delta = train_model(optimizer_ada_delta, model_ada_delta,"model_ada_delta")
            loss_adam,acc_adam,epoch_adam = train_model(optimizer_adam, model_adam,"model_adam")

            epochs = [epoch_delta,epoch_ada_delta,epoch_adam]
            

            print("\n \n Rule Delta",hid_dim1,hid_dim2,lr,loss_delta["val"][-1])
            print("\n \n Rule Ada Delta",hid_dim1,hid_dim2,lr,loss_ada_delta["val"][-1])
            print("\n \n Rule Adam",hid_dim1,hid_dim2,lr,loss_adam["val"][-1])
            csvwriter.writerow(["\n \n Rule Delta",hid_dim1,hid_dim2,lr,loss_delta["val"][-1]])
            csvwriter.writerow(["\n \n Rule Ada Delta",hid_dim1,hid_dim2,lr,loss_ada_delta["val"][-1]])
            csvwriter.writerow(["\n \n Rule Adam",hid_dim1,hid_dim2,lr,loss_adam["val"][-1]])

            """
            plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,"train",loss_or_accuracy="loss")
            plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,"val",loss_or_accuracy="loss")
            
            plot_comparative(acc_delta,acc_ada_delta,acc_adam,epochs,lr,"train",loss_or_accuracy="accuracy")
            plot_comparative(acc_delta,acc_ada_delta,acc_adam,epochs,lr,"val",loss_or_accuracy="accuracy")

            plot_confusion_matrix(lr,"model_delta.pth",train_dataloader)
            plot_confusion_matrix(lr,"model_ada_delta.pth",train_dataloader)
            plot_confusion_matrix(lr,"model_adam.pth",train_dataloader)

            plot_confusion_matrix(lr,"model_delta.pth",val_dataloader)
            plot_confusion_matrix(lr,"model_ada_delta.pth",val_dataloader)
            plot_confusion_matrix(lr,"model_adam.pth",val_dataloader)
            """





