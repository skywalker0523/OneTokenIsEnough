

python evaluate_diff.py --tasks lambada_standard --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='chain_rule',greddy=True,cfg=1.1

python evaluate_diff.py --tasks openbookqa,boolq --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='chain_rule',greddy=False,cfg=1.6

python evaluate_diff.py --tasks piqa --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='chain_rule',greddy=False,cfg=0.6

python evaluate_diff.py --tasks arc_easy --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='mc',greddy=False,cfg=0.6

python evaluate_diff.py --tasks social_iqa --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='mc',greddy=False,cfg=0.6

python evaluate_diff.py --tasks race --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='mc',greddy=False,cfg=0.7

python evaluate_diff.py --tasks hellaswag --model mdlm --batch_size 16 --model_args model_name=472,ckpt_path='your_model_save_path',nll_type='mc',greddy=False,cfg=0.7