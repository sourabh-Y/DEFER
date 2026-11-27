# DEFER: Detection with Encoder Features and Epistemic Rejection
 
Code for the paper **"DEFER: Detection with Encoder Features and Epistemic Rejection for LLM Hallucinations."**
 
DEFER treats hallucination detection as *selective classification*: for each source–response pair, it either returns a label with a probability and an uncertainty score, or it abstains so the case can be sent for further verification.
 
1. **Encoder features.** A frozen pretrained Transformer encoder maps the input to a fixed feature vector (computed once per input).
2. **Variational decision stage.** A two-layer Bayesian MLP learns a distribution over its weights (mean-field Gaussian, trained with a KL-regularized loss).
3. **Epistemic rejection.** At test time, the MLP is sampled `T = 20` times. The predictions are averaged, and the variance across samples is the uncertainty score. Inputs with high uncertainty are withheld.
Since the encoder runs only once, uncertainty estimation adds less than 1% to the inference cost. The method requires only the source text and the response, so it also works with outputs from closed-source LLMs.
 
## Repository contents
 
| File | Description |
|---|---|
| `notebook.ipynb` | Full pipeline on the Wiki-Hades: data loading, feature extraction, training, Monte Carlo inference, metrics, and accuracy–rejection curve |
| `README.md` | This file |
 
## Requirements
 
- Python 3.9+
- PyTorch
- Hugging Face `transformers` and `datasets`
- `scikit-learn`, `numpy`, `pandas`, `tqdm`, `matplotlib`
```bash
pip install torch transformers datasets scikit-learn numpy pandas tqdm matplotlib
```
 
A GPU is recommended for feature extraction. Training the decision stage is light and also runs on a CPU.
 
## Running the notebook
 
Open `notebook.ipynb` and run all cells in order. The notebook:
 
1. loads the dataset;
2. splits the data into train / validation / test samples with a fixed seed (42);
3. extracts the pooled features with a frozen `DeBERTa-v3-base` encoder;
4. trains the Bayesian decision stage for five random seeds;
5. runs 20 Monte Carlo passes on the test set and reports the metrics below, saving the accuracy–rejection curve data to CSV.
 
## License
**MIT**
