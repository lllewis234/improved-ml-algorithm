# Improved machine learning algorithm for predicting ground state properties


# Install

```bash
conda create -n imp_qml python=3.9
pip install -r requirements.txt
```

# how to use

* prepare dataset by executing `python dataloader.py`
* prepare kernels by executing `python kernels.py`
* to train the original methods (dirichlet,gauss,NTK kernel) execute `python train_kernel.py`
  * you can specify the test-set fraction or shadow-size or grid size as follows `python train_kernel.py --test-size 0.5 --shadow-size 500 --nrow 4`
  * you can find more options in `train.py@parse_args()`
* to train the new method execute `python train.py`
  * you can again specify test-set fraction or shadow size or grid size as above:  `python train.py --test-size 0.5 --shadow-size 500 --nrow 4`
  * to use the faster lasso library `celer` instead of `sklearn` execute:  `python train.py --test-size 0.5 --shadow-size 500 --nrow 4 --lasso-lib celer`
  * you can find more options in `train.py@parse_args()`

* Look into `Code_fast.ipynb` where we will use both methods to recreate the data to replicate the left plot in Fig.2