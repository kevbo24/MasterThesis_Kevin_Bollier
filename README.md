# MasterThesis_Kevin_Bollier

The project contains RL models based on algorithms such as DQN, A2C, PPO. It was examined to what extent these methods are suitable for a trading bot in the cryptocurrency sector and whether the integration of additional information sources such as Twitter tweets, Google Trends and the S&P500 stock index have a positive effect on the outcome.

# Execution of the source code

The desired method must be set for the executing scripts (PPO, A2C or DQN). It also requires the name of the script in which the algorithm is located. These can be found in the "src" folder.
![image](https://user-images.githubusercontent.com/36130935/171126298-cc8a9933-b8f7-46c4-8465-20e683df7a0f.png)

The storage of the model and the evaluations of the training requires a folder name (for training and validation). The number of time steps should also be specified.
![image](https://user-images.githubusercontent.com/36130935/171127162-7e425ab3-ad94-470e-aca7-059ccaf7a6b6.png)

Finally, the required files are defined for training and validation. These are located in the "data/data_processed_extended" folder for "untransformed data" and in the "data/data_processed_extended_stat" folder for the stationary data.
![image](https://user-images.githubusercontent.com/36130935/171127652-70a261e5-c2fe-4ae7-8e35-e9b09d30098a.png)

Note: A PPO model with inside information (backshift 7 days and backshift 24 hours) is based on the algorithm in the env_V36_PPO script. In addition, the correct data set has to be selected (df_processed_benchmark_backshift_7days or df_processed_benchmark_backshift_24). A different script name has to be used accordingly for A2C or DQN models.
