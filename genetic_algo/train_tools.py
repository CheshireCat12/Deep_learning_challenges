import numpy as np
import torch

class TrainHelper():

    def __init__(self, nb_epochs=20, device=torch.device('cpu')):
        self.max_epochs = nb_epochs
        self.device = device

    def train(self, model, train_loader, optimizer, loss_func):
        model.train()

        losses = []
        correct_train_pred = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Predict the classes of the model
            output = model(images)

            optimizer.zero_grad()

            # Compute the loss
            loss = loss_func(output, labels)

            # Perform backprop
            loss.backward()
            optimizer.step()

            # Save current loss
            losses.append(loss.item())

            # Save the number of correct classified items
            predicted_labels = output.argmax(dim=1)
            nb_correct = (predicted_labels == labels).sum().item()
            correct_train_pred += nb_correct

        train_accuracy = 100. * (correct_train_pred / len(train_loader.dataset))

        return np.mean(np.array(losses)), train_accuracy

    def validation(self, model, val_loader, loss_func):
        model.eval()

        losses = []
        correct_val_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = model(images)

                loss = loss_func(output, labels)

                # Save current loss
                losses.append(loss.item())

                # Save the number of correct classified items
                predicted_labels = output.argmax(dim=1)
                n_correct = (predicted_labels == labels).sum().item()
                correct_val_predictions += n_correct

        val_accuracy = 100. * (correct_val_predictions / len(val_loader.dataset))

        return np.mean(np.array(losses)), val_accuracy

    def _print_info(self, train_loss, val_loss, train_acc, val_acc):
        print(f'Train_loss: {train_loss:.3f} |\
                Val_loss: {val_loss:.3f} |\
                Train_acc: {train_acc:.3f} |\
                Val_acc: {val_acc:.3f}')

    def fit(self, model, train_loader, val_loader, optimizer, loss_func, scheduler=None,debug=False):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        ### Early stopping code
        best_val_loss = np.inf
        patience = 5 # if no improvement after 5 epochs, stop training
        counter = 0
        epochs_counter = 0

        for epoch in range(self.max_epochs):
            epochs_counter += 1

            train_loss, train_acc = self.train(model,
                                               train_loader,
                                               optimizer,
                                               loss_func)
            val_loss, val_acc = self.validation(model,
                                                val_loader,
                                                loss_func)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            ### Early stopping code
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
            ###
            if scheduler:
                scheduler.step()

            if debug:
                print(f'Epoch: {epoch+1}/{self.nb_epochs}')
                print_info(train_loss, val_loss, train_acc, val_acc)

        self._print_info(train_losses[-1], val_losses[-1], train_accuracies[-1], val_accuracies[-1])

        return train_losses, val_losses, train_accuracies, val_accuracies, epochs_counter
