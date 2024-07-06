***Please star this repo!!***

# 60-Days-of-learning / Deep learning PyTorch
## Day 1

### Topics Covered
- **What is Machine Learning?**
- **What Deep Learning is Not Good For?**
- **Neural Networks**
- **Types of Learning**
  - Supervised Learning
  - Unsupervised Learning (or Self-supervised Learning)
  - Reinforcement Learning
  - Transfer Learning
- **PyTorch**
- **Workflow of PyTorch**

## Day 2
## How to Approach This Course

### PyTorch Fundamentals

#### Introduction to Tensors
- **Random Tensors**
- **Zeros and Ones**
- **Creating a Range of Tensors and Tensors-like**

#### Tensor Datatypes
- **Data Types (dtype)**

#### Finding Details of a Tensor

#### Manipulating Tensors (Tensor Operations)
- **Mathematical Operations**
- **Reduction Operations**
  - sum
  - mean
  - max
  - min
  - product

#### Indexing and Slicing
- **Indexing**
- **Slicing**
- **Advanced Indexing**

#### Shape Manipulation
- **Reshape**
- **Transpose**
- **Concatenate**

## Day 3

### Topics Covered
- **Reshaping, Stacking, Squeezing, and Unsqueezing Tensors**
  - reshape
  - view
  - torch.stack()
  - torch.squeeze()
  - torch.unsqueeze
  - permute
- **PyTorch Tensors & NumPy**
- **Reproducibility (Trying to Take Random Out of Random)**
## Day 4

### Topics Covered
- **PyTorch Workflow**
  - Data (Preparing and Loading)
  - Linear Regression
  - Splitting Data into Training and Test Sets (One of the Most Important Concepts in Machine Learning in General)

  ## Day 5

### Topics Covered
- **Building Models**
- **Gradient Descent**
- **Checking the Contents of a PyTorch Model**
- **torch.inference_mode()**
- **Training Models**

## Day 6

### Topics Covered
- **Building a Training Loop (and a Testing Loop) in PyTorch**
- **Detailed Breakdown**
- **Saving a Model**
- **Loading model**

## Day 7

### Topics Covered
Day 7: Covered loading PyTorch models, creating device-agnostic code, preparing data, building, and training models. Putting all the pieces together! 🚀

## Day 8

### Topics Covered
- **Neural Network Classification**
- **Making Classification Data and Getting It Ready**
- **Making Dataframe of Circles**

## Day 9

### Topics Covered

Day 9: Checked input/output shapes, viewed the first sample, turned data into tensors, created train/test splits, built a model, made predictions, and used scikit-learn for train/test split. 🚀📊

## Day 10

### Topics Covered
- **Setup Loss Function and Optimizer**
- **Calculate Accuracy**
- **Train a Model**
- **Going from Raw Logits to Prediction Probabilities to Prediction Labels**
- **Using Sigmoid**
- **Building a Training and Testing Loop**

## Day 11

### Topics Covered
- **Make Predictions and Evaluate the Model**
- **Improving a Model (From a Model Perspective)**
- **Preparing Data to See if Our Model Can Fit a Straight Line**

### Day 12

### Topics Covered
 Adjusted model_1 to fit a straight line, conducted observations and analysis, evaluated model performance, identified possible improvements, addressed the missing piece: non-linearity, recreated non-linear data (red and blue circles), performed train/test split, and built a model with non-linearity. 🚀🔄

## Day 13

### Topics Covered
- **What is an Activation Function in Neural Networks?**
- **Building a Model with Non-Linearity**
- **Training a Model with Non-Linearity**
- **Evaluating a Model with Non-Linear Activation Functions**
- **Description of Evaluated Model**
- **Replicating Non-Linear Activation Functions**
- **Putting it All Together with a Multi-Class Classification**
- **Creating a Toy Multi-Class Dataset**

## Day 14

### Topics Covered
- **Building a Multi-Class Classification in PyTorch**
- **Creating a Loss Function and an Optimizer for a Multi-Class Classification Model**
- **Getting Prediction Probabilities for a Multi-Class PyTorch Model**
- **Converting Logits to Prediction Probabilities with Softmax**
- **Building Training and Testing Loops for a Multi-Class PyTorch Model**
- **Making and Evaluating Predictions with a PyTorch Multi-Class Model**

## Day 15

### Topics Covered
- **Computer Vision Networks**
- **Images**
- **Convolutional Neural Networks (CNN)**
- **Dataset**
- **Visualizing Data**
  
## Day 16

### Topics Covered
- **Preparing Dataloaders**
- **Model 0: Building a Baseline Model**
- **Setting Up Loss, Optimizer, and Evaluation Metrics**
- **Creating a Function to Time Experiments**

## Day 17

### Topics Covered
- **Creating a Training Loop and Training a Model on Batches of Data**
- **Making Predictions and Getting Model 0 Results**
- **Setting Up Device-Agnostic Code**

## Day 18

### Topics Covered
- **Model 1: Building a Better Model with Non-Linearity**
- **Setting Up Loss, Optimizer, and Evaluation Metrics**
- **Functionizing Training and Evaluation/Testing**
- **Train Loop Function**
- **Test Loop Function**

## Day 19

### Topics Covered
- **Get Model 1 Results**
- **Model 2: Building a Convolutional Neural Network (CNN)**

## Day 20
- **Explanation of the CNN model**
- **Stepping through `nn.Conv2d()`**
- **Stepping through `nn.MaxPool2d()`**

## Day 21
- **Set up a loss function and optimizer for Model 2**
- **Training and testing Model 2 using our training and test functions**
- **Get Model 2 results**
- **Compare model results and training time**
- **Make and evaluate random predictions with the best model**
- **Plot predictions**

## Day 22
- **Making a confusion matrix for further prediction evaluation**
- **Save and load the best performing model**

## **Day 23**
- **Learned about PyTorch Custom Datasets**
- **Imported PyTorch and set up device-agnostic code**
- **Retrieved data**
- **Prepared and explored data**
- **Visualized an image to understand the data better**

## **Day 24**
- **Explored transforming data with `torchvision.transforms`**
- **Learned how to load image data using `ImageFolder`**
- **Examined dataset details**

## **Day 25**
- **Turned loaded images into DataLoaders**
- **Explored loading image data with a custom dataset**
- **Created helper functions to get class names**

## **Day 26**
- **Created helper functions to get class names**
- **Built a custom Dataset to replicate `ImageFolder`**

## **Day 27**
- **Created a function to display random images**
- **Turned custom loaded images into DataLoaders**
- **Explored data augmentation**

## **Day 28**
- **Built Model 0: Tiny VGG without data augmentation**
- **Created transforms and loaded data for Model 0**
- **Defined the TinyVGG model class**
- **Tested the model with a forward pass on a single image**

## Day 29
**Use `torchinfo` to get an idea of the shapes going through our model**  
**Create train and test loop functions**  
**Creating a `train()` function to combine `train_step()` and `test_step()`**  
**Train and evaluate model 0**

## Day 30
**Plot the loss curves of Model 0**  
**What should an ideal loss curve look like?**  
**Model 1: TinyVGG with Data Augmentation**  
**Create transform with data augmentation**  
**Create train and test Datasets and DataLoader with data augmentation**  
**Construct and train model 1**  
**Plot the loss curves of model 1**  
**Compare model results**

## Day 31
**Making a prediction on a custom image**  
**Loading in a custom image with PyTorch**  
**Making a prediction on a custom image with a trained PyTorch model**  
**Putting custom image prediction together: building a function**

## Day 32
**Create Datasets and Dataloaders (script mode)**  
**Making a model (TinyVGG) with a script (`model_builder.py`)**  
**Turn training functions into a script (`engine.py`)**  
**Create a file called `utils.py` with utility functions**  
**Train, evaluate, and save the model (script mode) -> `train.py`**

## Day 33
**Transfer Learning Overview**  
**Why use Transfer Learning?**  
**What we are going to cover?**  
**Get data**  
**Create Datasets and DataLoaders**

## Day 34
**Create Datasets and DataLoaders**  
**Creating a transform for `torchvision.models` (manual creation)**  
**Creating a transform for `torchvision.models` (auto creation)**

## Day 35
**Getting a pretrained model**  
**Which pretrained model should you use?**  
**Setting up pretrained model**  
**Getting a summary of our model with `torchinfo.summary()`**  
**Freezing the base model and changing the output layer to suit our needs**  
**Train model**  
**Evaluate model by plotting curves**  
**Detailed analysis of Model Performance Curves**

## Day 36
**Make predictions on images from the test set**

