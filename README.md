# csv_pose_classification
The dataset is recorded some pose coordinates, I want to use it to classify the pose. 

My purpose is classified the pose in **model_train_pose.py** , which is basic modifying from **model_train_heart_disease.py**.  

In **model_train_pose.py** file, I get the some error message: **Unimplemented: Cast string to float is not supported**.  

I have no idea about this message means.  

Is that means my model.fit() input tensor fortmat is wrong?  

However, I can successful run the **model_train_heart_disease.py** file.  

Could any one help me to solve this problem? Thank a lot. 

![Screenshot from 2021-08-26 16-11-49](https://user-images.githubusercontent.com/19554347/130926531-8b5709cf-4b97-45bb-8ddd-44420b3adc96.png)

# Scripts 
**model_train_heart_disease.py** - This code is Modify from Keras documents tutorial: [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/), I learn how to use csv file to be classified.

**model_train_pose.py** - This file is what I want to classify the pose.

# Installation

**Conda virtual env**

```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install pandas==1.1.3
pip install numpy
pip install tensorflow-gpu==2.6.0
conda install cudnn==8.2.0.53
pip install pydot
```

# solutions: 

I find a solution to fix the issue.  

I chang the dataset label ('class') from 'string' to 'int' data type in numerical_coords_dataset.csv.  

And then model.fit() can start training.

![Screenshot from 2021-08-27 13-59-17](https://user-images.githubusercontent.com/19554347/131079383-4c96e398-8ecd-442b-bf74-2d465f783641.png)
