# From Peaks to Structures: Multi-Modal Structure Prediction via Learned Spectral Interpretation

Data availability:
All model checkpoints and datasets are stored at https://zenodo.org/records/17268577.

Model checkpoints:
best_struct_cp.pt -> regular model trained with MS + IR + H-NMR
best_struct_irnmr.pt -> regular model trained with IR + H-NMR
best_struct_irms.pt -> regular model trained with MS + IR
best_struct_ms.pt -> regular model trained with MS
best_struct_ir.pt -> regular model trained with IR
best_struct_nmr.pt -> regular model trained with H-NMR
best_rand_10(30,50,70)_struct_cp.pt -> random drop model trained with MS + IR + H-NMR but masked 10%(30%,50%,70%) of each spectral inputs (at least one spectral input will be guaranteed to stay)

Dataset:
train.zip -> preprocessed training data
val.zip -> preprocessed validation data
test.zip -> preprocessed test data
raw_spec1(2,3).zip -> preprocessed MS (m/z binned, intensity normalized), IR (cm-1 binned, intensity normalized), H-NMR (both shift and intensity binned). Please check binned and normalization details in our paper.

