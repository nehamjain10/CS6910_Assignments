import torch 
import torch.nn as nn
import numpy as np
import torchvision.models as models
from MLFFNN import MLFFNN
import matplotlib.pyplot as plt


def train_model(optimizer,criterion, model,train_dataloader,val_dataloader,MAX_EPOCHS,device="cpu"):
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
            torch.save(model, f'weights/best_model.pth')
            no_improvement=0
        else:
            no_improvement+=1
            if no_improvement==4:
                break
    if no_improvement>0:  
        loss_metrics["train"] = loss_metrics["train"][:-no_improvement]
        loss_metrics["val"] = loss_metrics["val"][:-no_improvement]
        
        accuracy_metrics["train"] = accuracy_metrics["train"][:-no_improvement]
        accuracy_metrics["val"] = accuracy_metrics["val"][:-no_improvement]
        
    return loss_metrics, accuracy_metrics,epoch-no_improvement


def plot_comparative(loss_delta,loss_ada_delta,loss_adam,epochs,lr,is_type="train",loss_or_accuracy="loss"):
    
    plt.figure(figsize=(10,8),dpi=300)
    print(epochs[0],epochs[1],epochs[2])
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


def plot_confusion_matrix(lr,model_type,data_type):
    if data_type=="train":
        dataloader = train_dataloader
    else:
        dataloader = test_dataloader
    model = torch.load(f'weights/{model_type}.pth')
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
        
        confusion_matrix = confusion_matrix.numpy()
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10,8),dpi=300)
        sns.heatmap(confusion_matrix, annot=True,fmt='.2f', xticklabels=idx_to_labels, yticklabels=idx_to_labels)  
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.title(f"Confusion Matrix with Learning Rate: {lr}")
        plt.savefig(f"figs/confusion_matrix_{model_type}_{data_type}_{lr}.png")