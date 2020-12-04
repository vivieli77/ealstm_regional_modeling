import subprocess
import glob, os
"""for i in range(1, 4):
    seed = str(i) * 5
    subprocess.run(["python", "main.py", "train", "--camels_root", "data", "--seed", seed, "--cache_data", "True", "--num_workers", "4", "--use_mse", "True"])
for i in range(1, 4):
    seed = str(i) * 5    
    subprocess.run(["python", "main.py", "evaluate", "--camels_root", "data", "--run_dir", "runs/run_seqlen14_seed" + seed])"""
"""for i in range(1, 10):
    seed = str(i) * 5
    subprocess.run(["python", "main.py", "train", "--camels_root", "data", "--seed", seed, "--cache_data", "True", "--num_workers", "4", "--use_mse", "True"])
for i in range(1, 10):
    seed = str(i) * 5    
    subprocess.run(["python", "main.py", "evaluate", "--camels_root", "data", "--run_dir", "runs/run_seqlen14_seed" + seed])"""
seed = "11111"
# finding the perfect seqlen!
"""subprocess.run(["python", "main.py", "train", "--camels_root", "data_mob", "--seed", seed, "--cache_data", "True", "--num_workers", "4", "--use_mse", "True"])
subprocess.run(["python", "main.py", "evaluate", "--camels_root", "data_mob", "--run_dir", "runs/run_seqlen30_seed" + seed])"""
"""hidden_states = [64, 128, 192, 256]
dropout_rates = [0.05, 0.1, 0.2, 0.4]
for h in hidden_states:
    for d in dropout_rates:
        print(str(h) + str(d))
        subprocess.run(["python", "main.py", "train", "--camels_root", "data_mob", "--seed", seed, "--cache_data", "True", "--use_mse", "True", "--hidden_size", str(h), "--dropout", str(d)])"""
#mobility vs. without mobility data
"""for i in range(1, 10):
    seed = str(i) * 5 
    #subprocess.run(["python", "main.py", "train", "--camels_root", "data_mob", "--seed", seed, "--cache_data", "True", "--use_mse", "True", "--hidden_size", "192", "--dropout", "0.1"])
    #print("Now training on data without mobility variable:")
    subprocess.run(["python", "main.py", "train", "--camels_root", "data", "--seed", seed, "--cache_data", "True", "--use_mse", "True", "--hidden_size", "192", "--dropout", "0.1"])
print("Now evaluating all runs:")"""
"""for file in glob.glob("runs/run*"):
    print(file)
    subprocess.run(["python", "main.py", "evaluate", "--camels_root", "data", "--run_dir", file])"""

for i in range(1, 10):
    seed = str(i) * 5 
    subprocess.run(["python", "main.py", "train", "--camels_root", "data_mob", "--seed", seed, "--cache_data", "True", "--no_static", "True", "--use_mse", "True", "--hidden_size", "192", "--dropout", "0.1"])
for file in glob.glob("runs/run*"):
    print(file)
    subprocess.run(["python", "main.py", "evaluate", "--camels_root", "data_mob", "--run_dir", file])