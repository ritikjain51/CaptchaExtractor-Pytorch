from tqdm import tqdm
import torch
import config


def train_fn(model, dataloder, optimizer):

    model.train()
    fin_loss = 0
    tk = tqdm(dataloder, total = len(dataloder))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    
    return fin_loss/len(dataloder)

def eval_fn(model, dataloder, optimizer):

    model.train()
    fin_loss = 0
    fin_pred = []
    tk = tqdm(dataloder, total = len(dataloder))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        batch_pred, loss = model(**data)
        fin_loss += loss.item()
        fin_pred.append(batch_pred)


    return fin_pred, fin_loss/len(dataloder)

    