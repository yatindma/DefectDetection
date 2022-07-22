import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from sklearn.metrics import multilabel_confusion_matrix

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        with t.set_grad_enabled(True):
            # -propagate through the network
            model_out = self._model(x)
            # -calculate the loss
            loss = self.crit(model_out, y)  # loss function
            # -compute gradient by backward propagation
            loss.backward()
            # -update weights
            self._optim.step()
            # -return the loss
            return loss

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        model_output = self._model(x)
        val_test_loss = self._crit(model_output, y)
        # return the loss and the predictions
        return val_test_loss, model_output

    def train_epoch(self):
        # set training mode
        self._model.train()
        batch_loss = 0.0
        batch_count = 0
        # bring everything to device
        # iterate through the training set
        index = 0
        loss = []
        f1_scores = []
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        for images, label in self._train_dl:
            index += 1
            images = images.to(self.device)
            labels = label.to(self.device)
            loss.append(self.train_step(images, labels))
            # calculate the average loss for the epoch and return it
            average_loss = t.mean(t.tensor(loss))
            # print('Epoch {:02} | Batch {:03}-{:03} | Train loss: {:.3f}'.
            #           format(self.epoch, previous_idx, index, average_loss))
            previous_idx = index

        return average_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        self._model.eval()
        # iterate through the validation set
        losses = []
        f1_scores = []
        with t.no_grad(): # so that back propagation can be stopped
            for images, label in tqdm(self._val_test_dl):
                # transfer the batch to the gpu if given
                images = images.to(self.device)
                labels = label.to(self.device)
                # perform a validation step
                val_test_loss, model_output = self.val_test_step(images, labels)
                losses.append(val_test_loss)

                f1_scores.append(f1_score(int(model_output > 0.5), labels))
        # save the predictions and the labels for each batch
        print(f"F1 score is: {t.mean(t.tensor(f1_scores, dtype=t.float32))}")
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        average_loss = t.mean(t.tensor(losses))
        # return the loss and print the calculated metrics
        return average_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        loss_arr = []
        loss_train= []
        loss_val = []
        while True:
            # stop by epoch number
            if epochs <= 0:
                break
            epochs -= 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss = self.val_test()

            # append the losses to the respective lists
            loss_train.append(train_loss)
            loss_val.append(val_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epochs)

            loss_arr.append(train_loss)


            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # TODO

            # return the losses for both training and validation
        return loss_train, loss_val




