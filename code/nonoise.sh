# no noise
python main.py --model_type n2v --num_neg_samples_score 50 --num_neg_samples_fit 50 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type n2v --num_neg_samples_score 500 --num_neg_samples_fit 500 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type n2v --num_neg_samples_score 1000 --num_neg_samples_fit 1000 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type n2v --num_neg_samples_score 2000 --num_neg_samples_fit 2000 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type n2v --num_neg_samples_score 5000 --num_neg_samples_fit 5000 --n_missing_edges 0 --n_spurious_edges 0

python main.py --model_type s2v --num_neg_samples_score 50 --num_neg_samples_fit 50 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type s2v --num_neg_samples_score 500 --num_neg_samples_fit 500 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type s2v --num_neg_samples_score 1000 --num_neg_samples_fit 1000 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type s2v --num_neg_samples_score 2000 --num_neg_samples_fit 2000 --n_missing_edges 0 --n_spurious_edges 0
python main.py --model_type s2v --num_neg_samples_score 5000 --num_neg_samples_fit 5000 --n_missing_edges 0 --n_spurious_edges 0

# with 1% noise
python main.py --model_type n2v --num_neg_samples_score 50 --num_neg_samples_fit 50 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type n2v --num_neg_samples_score 500 --num_neg_samples_fit 500 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type n2v --num_neg_samples_score 1000 --num_neg_samples_fit 1000 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type n2v --num_neg_samples_score 2000 --num_neg_samples_fit 2000 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type n2v --num_neg_samples_score 5000 --num_neg_samples_fit 5000 --n_missing_edges 1 --n_spurious_edges 1spurious_edges 0

python main.py --model_type s2v --num_neg_samples_score 50 --num_neg_samples_fit 50 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type s2v --num_neg_samples_score 500 --num_neg_samples_fit 500 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type s2v --num_neg_samples_score 1000 --num_neg_samples_fit 1000 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type s2v --num_neg_samples_score 2000 --num_neg_samples_fit 2000 --n_missing_edges 1 --n_spurious_edges 1
python main.py --model_type s2v --num_neg_samples_score 5000 --num_neg_samples_fit 5000 --n_missing_edges 1 --n_spurious_edges 1