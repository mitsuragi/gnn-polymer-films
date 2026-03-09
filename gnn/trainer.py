import torch
from torch_geometric.loader import DataLoader

def train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device=None
):
    model.train()

    total_loss = 0

    all_preds = []
    all_targets = []
    all_probs = []

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch).squeeze()

        target = data.y.float().view(-1)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        probs = torch.sigmoid(out)
        preds = (probs>0.1).int()

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)

    # regression-like metrics
    mae = torch.mean(torch.abs(probs - targets))

    rmse = torch.sqrt(torch.mean((probs - targets) ** 2))

    wape = torch.sum(torch.abs(probs - targets)) / torch.sum(torch.abs(targets) + 1e-8)

    # classification metrics
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "WAPE": wape.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
    }

    return metrics


def eval(
    model,
    loader,
    criterion,
    device=None
):
    model.eval()

    total_loss = 0 
    
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch).view(-1)

            target = data.y.float().view(-1)

            loss = criterion(out, target)

            total_loss += loss.item() * data.num_graphs

            probs = torch.sigmoid(out)
            preds = (probs>0.5).int()

            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_targets.append(target.detach().cpu())

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)

    # regression-like metrics
    mae = torch.mean(torch.abs(probs - targets))

    rmse = torch.sqrt(torch.mean((probs - targets) ** 2))

    wape = torch.sum(torch.abs(probs - targets)) / torch.sum(torch.abs(targets) + 1e-8)

    # classification metrics
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "WAPE": wape.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
    }

    return metrics


def test(
    model, loader, device=None
):
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch).view(-1)

            target = data.y.float().view(-1)

            probs = torch.sigmoid(out)
            preds = (probs>0.1).int()

            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_targets.append(target.detach().cpu())

    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)

    # regression-like metrics
    mae = torch.mean(torch.abs(probs - targets))

    rmse = torch.sqrt(torch.mean((probs - targets) ** 2))

    wape = torch.sum(torch.abs(probs - targets)) / torch.sum(torch.abs(targets) + 1e-8)

    # classification metrics
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    metrics = {
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "WAPE": wape.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
    }

    return metrics
