import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torchvision.transforms as transforms
import imageio

def plot_confusion_matrix(lr,model_type,data_type,dataloader,device,idx_to_labels):

    model = torch.load(f'weights/{model_type}.pth', map_location=device)
    model.eval()
    with torch.no_grad():
        confusion_matrix = torch.zeros(5, 5)
        for X_test, y_test in dataloader:
            X_test = X_test.type(torch.float32).to(device)
            y_test = y_test.type(torch.long).to(device)
            score = model(X_test)
            score = score.squeeze(dim=1)
            #y_test = y_test.squeeze(dim=1)
            max_indices = torch.argmax(score, dim=1)
            for t, p in zip(y_test.cpu(), max_indices.cpu()):
                confusion_matrix[t.long(), p.long()] += 1
        
        confusion_matrix = confusion_matrix.numpy()
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10,8),dpi=300)
        sns.heatmap(confusion_matrix, annot=True,fmt='.2f', xticklabels=idx_to_labels, yticklabels=idx_to_labels)  
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.title(f"Confusion Matrix for best performing model")
        plt.savefig(f"figs/confusion_matrix_{model_type}_{data_type}.png")

    
def train_model(optimizer,criterion, model,train_dataloader,val_dataloader,MAX_EPOCHS,device="cpu",save_name = "best_model"):
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
    t = time.time()
    for epoch in range(MAX_EPOCHS):
        train_loss = []
        t = time.time()
        correct_classified = 0
        count = 0
        for X_train, y_train in train_dataloader:
            count+=1
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.long).to(device)

            optimizer.zero_grad()

            score = model(X_train)
            score = score.squeeze(dim=1)
            #y_train = y_train.squeeze(dim=1)
            
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
                
                #y_valid = y_valid.squeeze(dim=1)
                
                loss = criterion(input=score, target=y_valid)
                validation_loss.append(loss.item())

                max_indices = torch.argmax(score, dim=1)
                correct_classified +=  (max_indices == y_valid).float().sum()
        
        accuracy_metrics["val"].append(correct_classified.item() / len(val_dataloader.dataset))
        
        loss_metrics["train"].append(np.average(train_loss))
        loss_metrics["val"].append(np.average(validation_loss))

        if np.average(validation_loss) < best_loss:
            best_loss = np.average(validation_loss)
            torch.save(model, f'weights/{save_name}.pth')
            no_improvement=0
        else:
            no_improvement+=1
            if no_improvement==4:
                break

    if no_improvement>0:  
        loss_metrics["train"] = loss_metrics["train"]#[:-no_improvement]
        loss_metrics["val"] = loss_metrics["val"]#[:-no_improvement]
        
        accuracy_metrics["train"] = accuracy_metrics["train"]#[:-no_improvement]
        accuracy_metrics["val"] = accuracy_metrics["val"]#[:-no_improvement]
        
    return loss_metrics, accuracy_metrics,epoch#-no_improvement


def plot_comparative(metric,loss_or_accuracy="loss",model_type="CNN"):
    
    plt.figure(figsize=(10,8),dpi=300)
    plt.plot(range(1,len(metric["train"])+1),metric["train"],label="train")
    plt.plot(range(1,len(metric["val"])+1),metric["val"],label="validation")
    plt.legend()
    plt.title(f"Training {loss_or_accuracy} vs Number of epochs")
    plt.savefig(f"figs/training_{loss_or_accuracy}_{model_type}.png")



def plot_misclassified_examples(model_type,dataloader,device,idx_to_labels):
    model = torch.load(f'weights/{model_type}.pth', map_location=device)
    model.eval()

    model.eval()

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
    )
    with torch.no_grad():
        count = 0
        for X_test, y_test in dataloader:
            X_test = X_test.type(torch.float32).to(device)
            y_test = y_test.type(torch.long).to(device)
            score = model(X_test)
            X_test = inv_normalize(X_test)
            #score = score.squeeze(dim=1)
            pred = score.argmax(dim=1, keepdim=True) #pred will be a 2d tensor of shape [batch_size,1]
            for i in range(len(pred)):
                if(pred[i]!=y_test[i]): 
                    imageio.imwrite(f"misclassified_examples/{idx_to_labels[y_test[i].squeeze().cpu().numpy()]}_{idx_to_labels[pred[i].squeeze().cpu().numpy()]}_{count}.jpg",np.moveaxis(X_test[i].squeeze().cpu().numpy(),0,-1))
                    count+=1
