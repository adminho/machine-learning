# Example: Neural network 2 layers
## Descriptions

Use AI to calculate the logic `X1 or X1 xor X3`

![dataset_logic.PNG](images/dataset_logic.PNG)

In [__'tensorflow_2_layer.py'__](tensorflow_2_layer.py), you can edit the root path and dataset file name

```
csv_dir = 'D:/MyProject/machine-learning/Neural network' 	# your root path of a the dataset file		
df = pd.read_csv(os.path.join(csv_dir, 'example_2_layer.csv'), dtype=np.float32) 
```
