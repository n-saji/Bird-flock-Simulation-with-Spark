import os
os.system('clear')
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes
    
@pandas_udf(IntegerType())
def MLPClassifier_udf(*batch_inputs):
    batchTensor = torch.tensor([list(row) for row in zip(*batch_inputs)], dtype=torch.float32)
    with torch.no_grad():
        predictions = mlp_model(batchTensor)
    return pd.Series(predictions.numpy())


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of sentences")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="hidden_dim")
    parser.add_argument('--hidden_layer', type=int, default=50, help="hidden_layer")
    args = parser.parse_args()

    input_dim = 128 
    num_classes = 10 
    hidden_dims = [args.hidden_dim * args.hidden_layer]

    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    x = torch.randn(args.n_input, input_dim)  

    # Spark version
    spark = SparkSession.builder.appName("MLPClassifierDistributed").getOrCreate()
    df = pd.DataFrame(x.numpy(), columns=[f"feature_{i}" for i in range(x.shape[1])])
    spark_df = spark.createDataFrame(df)
    start_time = time.time()
    resultData = spark_df.withColumn("prediction", MLPClassifier_udf(*[spark_df[col] for col in spark_df.columns]))
    end_time = time.time()
    time1 = end_time - start_time
    spark.stop()
    print(f"Time taken for distributed classification: {end_time - start_time:.6f} seconds")

    # Non-spark version
    start_time = time.time()
    output = mlp_model(x)
    end_time = time.time()
    time2 = end_time - start_time
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")

    print(f"Time cost for spark and non-spark version: [{time1:.3f},  {time2:.3f}] seconds")
