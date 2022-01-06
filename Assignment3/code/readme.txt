Required Packages:
==================

1.torch
2.torchvision
3.torch.utils.model_zoo
4.torch.nn
5.torchvision.datasets
6.torchvision.transforms as transforms
7.torch.optim as optim

Description:
============

cifar_finetune.py

(
If the code does not run in local, I have run the same code in google colab
https://colab.research.google.com/drive/17fqCN0DAFvdYthiPpY7PZKUHoKTU-H4p?usp=sharing
)

For 3rd, 6th and 10th epoch, the code validates the CIFAR10 test dataset and stores the testing error and
prints the accuracy scores.

Function:- main() (where training is done)
    1. The given model resnet20() is initialized
    2. Checked if GPU is available, if so make the model to use GPU
    3. Data is converted into tensors, and normalize each channel of the image (RGB - Red,Green,Blue) with mean
       0.4914, 0.4822, 0.4465 and standard deviation of 0.2023, 0.1994, 0.2010
    4. The train and test datasets are downloaded from the CIFAR10 dataset provided by pytorch.
    5. Both train and test data are loaded using data loader and shuffled and batched with batch size of 32
    6. The loss function is defined.
    7. Then SGD optimizer is used to optimize the parameters in the fully connected layers(fc1 and fc2), with learning
       rate of 0.001 and momentum of 0.9 and weight decay(L2 regularisation 1e-3)
    8. After this we iterate the training dataset for 10 epochs and train the model, also check if CUDA is available and
       sets the images,labels to cuda
    9. Calculate the loss using the loss criterion(criterion = nn.CrossEntropyLoss()) and backpropagate the loss to
       adjust the weights and biases, adjust the optimizer for parameters and calculate running loss for training data.
    9. At the 3rd, 6th and 10th epoch, we call the test method to calculate accuracy and loss
    10.Based on the model that has the best validation error, we save the model as 'best_model.pth'

Function:- test()
    1. We pass the model, testing data loader, loss calculation criterion
    2. With no grad(making the gradients to zero), we iterate over the test data.
    3. Check if CUDA is available and sets the images,labels to cuda
    4. Calculate the loss using the loss criterion(criterion = nn.CrossEntropyLoss())
    5. Calculate the sample size and the correct predictions, then calculate accuracy and loss values


Run:
====
1. Open cmd/terminal and set the current working directory to the directory where the python file is present.
2. Type python cifar_finetune.py

Explanations:

Question1:
Run validation of the model every few training epochs on validation or test set of the dataset and save the model with
the best validation error.

+-------------------------+------------------+-----------------+------------------+
|          usecase        |            Epoch3|           Epoch6|           Epoch10|
+-------------------------+------------------+-----------------+------------------+
|Without L2 regularisation| Accuracy: 65.48 %|Accuracy: 65.13 %| Accuracy: 65.11 %|
|                         |     Loss: 0.99547|    Loss: 0.98087|     Loss: 0.98185|
+-------------------------+------------------+-----------------+------------------+
|  With L2 regularisation | Accuracy: 64.96 %|Accuracy: 65.11 %| Accuracy: 65.35 %|
|                         |     Loss: 0.99961|    Loss: 0.98046|     Loss: 0.97572|
+-------------------------+------------------+-----------------+------------------+
|   With 2 fc layers      | Accuracy: 65.26 %|Accuracy: 65.28 %| Accuracy: 65.06 %|
| (fully cconnected layer)|     Loss: 0.98340|    Loss: 0.97905|     Loss: 0.97988|
+-------------------------+------------------+-----------------+------------------+

In all the 3 cases, the best model turned out to be the the one produced after 10 epochs of training with the lowest
testing error. The accuracy seemed to be around ~65% for most scenarios. The model could have been trained better if
it has been trained on weights as well rather than the fc parameters and the no of epochs had been increased.

I also tried with 2 fully connected layers(fc1 and fc2). hoping to get a better accuracy and noted the values in the
table above.

I have also tried with various learning rates and 0.001 was the learning rate which gave me best accuracy.

Question2:
Try applying L2 regularization to the coefficients in the small networks we added.

After applying L2 regularization(implemented through weight decay), the accuracy seemed to be better than the one
without regularisation. I also tried to implement various weight decay values such as 1e-2, 1e-3, 1e-4, 1e-5. Among
those 1e-3 seemed to have performed the better. The following table outlines the same, all these are with 0.001 learning
rate and with 2 fully connected layers.

+-------------------------+------------------+-----------------+------------------+
|          L2 values      |            Epoch3|           Epoch6|           Epoch10|
+-------------------------+------------------+-----------------+------------------+
|           1e-2          | Accuracy: 65.13 %|Accuracy: 64.77 %| Accuracy: 65.23 %|
|                         |     Loss: 0.99607|    Loss: 0.99665|     Loss: 0.99324|
+-------------------------+------------------+-----------------+------------------+
|           1e-3          | Accuracy: 64.96 %|Accuracy: 65.11 %| Accuracy: 65.35 %|
|                         |     Loss: 0.99961|    Loss: 0.98046|     Loss: 0.97572|
+-------------------------+------------------+-----------------+------------------+
|           1e-4          | Accuracy: 64.80 %|Accuracy: 65.09 %| Accuracy: 65.32 %|
|                         |     Loss: 0.99427|    Loss: 0.97626|     Loss: 0.98013|
+-------------------------+------------------+-----------------+------------------+
|           1e-5          | Accuracy: 64.65 %|Accuracy: 65.19 %| Accuracy: 64.70 %|
|                         |     Loss: 0.99488|    Loss: 0.98694|     Loss: 0.97876|
+-------------------------+------------------+-----------------+------------------+

