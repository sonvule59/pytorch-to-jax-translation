import jax
import jax.numpy as jnp
from jax import random
from jax.vmap import vmap
from jax.dataframe import DataFrame
import json

def main():
    #Loading dataset
    with open('app/datasets/action_dataset.json') as f:
        data = json.load(f)
    df = DataFrame(data)
    
    #Setup model and data pipeline
    #Assuming action_models is defined in a JAX-compatible way
    action_train(df, 20000)  #Training the model with dataframe
    input_text = "tell me more about RIT"
    
    #Making prediction
    prediction = action_predict(input_text)  
    print(prediction)  

if __name__ == "__main__":
    jax.init precaution_check=True
    main()