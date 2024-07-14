Workflow of the Training Model Progress
## Initialization:

    The SCP_Experiment class is initialized with the experiment name, task, data folder, output folder, and list of models.

    The folder structure for the experiment is created if it does not already exist.

## Data Preparation:

    The dataset is loaded from the specified data folder.

    Label data is preprocessed and relevant data is selected and converted to one-hot encoding.

    The data is split into training, validation, and test sets based on the specified folds.

    Signal data is preprocessed.

    Training and test labels are saved.

## Model Training:

    For each model specified in the experiment:

    A folder for the model outputs is created.

    The respective model is loaded.

    The model is trained on the training data.

    Predictions are made on the training, validation, and test sets and saved.

## Model Evaluation:

    Labels for training, validation, and test sets are loaded.

    Bootstrap samples are generated if required.

    For each model, performance is evaluated using bootstrapping or point estimates.

    Evaluation results are saved.

## Execution:

    The main function in main.py defines the data and output folders, models, and experiments.

    For each experiment, the data is prepared, models are trained, and optionally evaluated.