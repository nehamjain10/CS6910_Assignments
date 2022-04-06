import torch 
import numpy as np


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
            print(y_train.shape,score.shape)
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
        
        print(np.average(train_loss))
        print(np.average(validation_loss))
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

