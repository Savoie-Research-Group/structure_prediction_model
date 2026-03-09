This readme file is for all evaluation tasks we did on our structure prediction model

Environment: pytorch=1.13.0, python=3.7, rdkit=2020.09.1.0

Note:
sys.path.append('/depot/bsavoie/data/Tianfan/structure_paper/train_script') -> the appended dir should be changed to the dir of train_script

1. Regular testing [test_script.py only counts the number of correct predictions. If needing to output smiles, use test_smiles.py instead]

   python test_script.py -c config_regular.txt

2. Random drop testing [A certain percent of each spectral sources are randomly dropped in test input]

   python test_script_random_drop.py -c config_regular.txt

3. Test random drop model on limited spectal sources inputs

   python test_ir/irnmr/nmr/ms_rand.py -c config_regular.txt

4. Test model performance when contradictory results are inputted

   python test_contra.py -c config_contra.txt

5. Output the per-token decisiveness on tested smiles

   python test_decisive.py -c config_decisive.txt 
