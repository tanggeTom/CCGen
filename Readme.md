## Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock

### To run the model

1. Modify the relative paths of the dataset in lines 64 and 65 of `run.py`.
2. Update the `model_dir` in `run.sh` to the output path, preferably named the same as the dataset.
3. Execute the script `nohup ./run.sh > xxx.log 2>&1 &` after making these changes in `run.sh`.
4. After `run.sh` completes, modify the `model_dir` variable in `test.sh` to the directory corresponding to the model.
5. Use `nohup ./test.sh > xxx.log 2>&1 &` to execute the `test.sh` script.
6. Running `remove_index_for_data_test` will remove indices from the output.
7. Execute `lowerAlpha.py` to convert the output to lowercase.
8. `metrics.py` is the evaluation script, providing various BLEU scores.
