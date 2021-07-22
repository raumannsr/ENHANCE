The 1_code folder contains the baseline and multi-task models used based on three different CNN architectures. Here is a description of the files:

- 0_baseline.py: The baseline model predicts a binary label (malignant or not) from a skin lesion image. The model is built on a convolutional base and extended further by adding specific layers. As an encoder, we used the VGG16, Inception v3, and ResNet50 convolutional base.

- 1_multi_task.py: The multi-task model extended the VGG16, Inception v3, and ResNet50 convolutional base with three fully connected layers. The model has two outputs with different network heads: one head is the classification output (abnormal or healty), the other represents the visual characteristic.

- 2_ensembles.py: We based the ensembles on the predictions of three available multi-task models: asymmetry, border and color. Per annotation source (student, crowd and automated), we calculate using averaging ensemble technique the class prediction (abnormal or healthy).

- 5_asymmetry.py: Code for calculating the asymmetry score based on existing ISIC and PH2 binary masks.
