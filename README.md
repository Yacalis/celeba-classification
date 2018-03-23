# cs274c-project

## Purpose:
Project for Neural Networks and Deep Learning (CS274C) class at UC Irvine.

## Topic:
The project is to look at how changing the batch size, instead of learning rate, will affect training a neural network.

## Null Hypothesis:
Increasing the batch size by alpha in plateauing regions of validation loss performs the same or worse (in terms of final accuracy over a similar number of epochs and otherwise identical conditions) than decreasing the learning rate by alpha during training.

## Process:
(1) Obtain dataset that has images of faces, and accompanying labels
(2) Create NN in Keras similar to existing human-face-classification NNs
(3) Rework NN to allow for batch size to change as training progresses
(4) Train different NNs on data with varying hyperparameters
(5) Create iOS app to take a saved Keras model and apply it to images taken by the phone
(6) Present work

## Dataset and Labels:
Dataset: CelebA - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Labels: binary - male, smiling, eyeglasses

## Initial Results:
The null hypothesis is true, and batch size increase performs worse than learning rate decrease during training. The most complicated networks we tried, which were based off of previous papers, did not perform as well as simpler networks. Overall, accuracy was around 93% for the three labels we tried. Accuracy was determined by the average correct prediction rate across all labels.

## Future Tests:
Need to try a non-Adam based optimizer, which could be confounding the results, as it has inherent decay. Also, can try different initial weights to see if they work better with more complicated models. Lastly, need to change iOS code to better size and preproccess images for input into the model -- currently, the resizing is poor enough to where the model has terrible accuracy.

## Relevent Papers:
Levi, G., & Hassner, T. (2015). Age and gender classification using convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 34-42).

Devarakonda, A., Naumov, M., & Garland, M. (2017). AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks. arXiv preprint arXiv:1712.02029.

Smith, S. L., Kindermans, P. J., & Le, Q. V. (2017). Don't Decay the Learning Rate, Increase the Batch Size. arXiv preprint arXiv:1711.00489.

Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision (pp. 3730-3738).

## Note on iOS App Title:
Originally, the dataset was based on the imdb celebrity faces dataset, and we only predicted a binary male classifier. However, the dataset had a very large number of unusable images, and remaining images were approximately 15-20% mislabeled. The original app design was based on work perform on this dataset, and the title has not changed even though the experiment has changed.
