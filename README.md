# SVM-ClassificationSpamEmail

- This project aims to using Support Vector Machine (SVM) to fix a classification problem which is classifying an email message as spam message or not. Through this project, we can go deeper about principles of SVMs and learn some practical and helpful data processing methods. All computer experiments in this projects are conducted in MATLAB.

- The dataset used in this project is from Spambase Data Set1 in Machine Learning Repository of UCI. It consists of 4,601 examples and each one of them contains a vector containing 57 attributes considered to be the key features of an email, and a scalar (1 or -1) which is regarded as the label of an email message. One means the email message is spam, while minus one means that the email is not spam. All attributes and labels are real number.

- There are three subsets of Spambase Data Set used in this computer experiment. The first one is the training set containing 2,000 examples, then the test set containing 1,536 examples, and finally the evaluation set used for making assessment for SVM. One important thing need to be noticed is that each example in evaluation set only have 57 attributes and do not have a label, this set is used to make application of a trained SVM with relatively good performance.

#### Task

Task one is to train a SVM. 
We apply different kernels and different type of margin in SVM to find an optimal hyperplane. This step can be regarded as constructing a SVM. We need to construct SVMs under all kinds of situations with different kernels and margins in this stage. 

Task two is to test a trained SVM. In this step, we need to feed both training data and test data into a trained SVM in task one, and then compute and record SVMs’ classification accuracy under all situations. 

Task three is to choose one among all trained SVMs as the target SVM. Build up a complete SVM with proper input and output function which is totally ready for application.



