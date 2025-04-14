def train(epoch, model, train_loader, criterion, optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    size = len(train_loader.dataset)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)


         # Verifica su quale dispositivo sono il modello e i dati
        #print(f"Model device: {next(model.parameters()).device}")
        #print(f"Inputs device: {inputs.device}")
        #print(f"Targets device: {targets.device}")

        # todo...
        # Zero the parameter gradients
        optimizer.zero_grad()

        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs,targets)

        #backpropagatiom
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
          current = batch_idx * inputs.size(0)
          print(f"loss: {running_loss:>7f} [{current:>5d}/{size:>5d}]")

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    return train_loss, train_accuracy
