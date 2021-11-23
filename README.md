## RBG for VRP

#### Dependencies

- Python>=3.6
- NumPy
- SciPy
- [PyTorch](http://pytorch.org/)
- tqdm

------

#### Evaluation

The models already trained is located in pretrained\ folder, and the dataset for evaluation is in data\ folder. You can directly evaluate the trained model on the evaluation dataset using the following code:

```sh
# to evaluate the trained model, run the following:

## --load_model is the address for the saved model, --size is the problem size, in [500,1000,2000].

## N=500 
python eval.py --size 500 --iter_step_eval 100 --eval_batch_size 100 --load_model pretrained/model500.pkl

## N=1000
python eval.py --size 1000 --iter_step_eval 100 --eval_batch_size 100 --load_model pretrained/model1000.pkl 

## N=2000
python eval.py --size 2000 --iter_step_eval 100 --eval_batch_size 50 --load_model pretrained/model2000.pkl
```

-----

#### Generate evaluation dataset

The dataset for evaluation is in data\ folder, which can be directly used. You can also generate the same dataset using the following code: (change --size to specify the problem size of the dataset)

```
python data_generate.py --size 500
```

-----

#### Pretrain

The pre-trained model is already in pretrained\ folder. You can also run the following code to get the pre-trained model.

```sh
python pretrain.py
```

-----

#### Train the model

To train the model, run the following:

```python

## N=500
python region_vrp.py --size 500 --epoch_size 100 --batch_size 14 --seed 200

## N=1000
python region_vrp.py --size 1000 --epoch_size 50 --batch_size 6 --seed 200

## N=2000
python region_vrp.py --size 2000 --epoch_size 25 --batch_size 2 --seed 200

```

