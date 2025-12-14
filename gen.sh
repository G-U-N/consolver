python gen_ppo.py --num_inference_steps 5  --cfg=3  --generation_path="baselines/0420_subset/ours/4order_depth/cfg3/5step/3000/"  --personalized_path="outputs/depth_4order_cfg3_prod_num_action_21_test_input_naive/checkpoint-3000/model.ckpt" --scaler_dim=0 --order_dim=4 --factor_num_actions=21
python gen_ppo.py --num_inference_steps 8  --cfg=3  --generation_path="baselines/0420_subset/ours/4order_depth/cfg3/8step/3000/"  --personalized_path="outputs/depth_4order_cfg3_prod_num_action_21_test_input_naive/checkpoint-3000/model.ckpt" --scaler_dim=0 --order_dim=4 --factor_num_actions=21
python gen_ppo.py --num_inference_steps 10 --cfg=3  --generation_path="baselines/0420_subset/ours/4order_depth/cfg3/10step/3000/" --personalized_path="outputs/depth_4order_cfg3_prod_num_action_21_test_input_naive/checkpoint-3000/model.ckpt" --scaler_dim=0 --order_dim=4 --factor_num_actions=21


python gen_ppo.py --num_inference_steps 1 --cfg=1  --generation_path="baselines/1101_subset/dmd/cfg3/1step/"



python gen_ppo.py --num_inference_steps 5 --cfg=3   --generation_path="baselines/1101_subset/deis/cfg3/5step/"  --type="deis"
python gen_ppo.py --num_inference_steps 8 --cfg=3   --generation_path="baselines/1101_subset/deis/cfg3/8step/"  --type="deis" 
python gen_ppo.py --num_inference_steps 10 --cfg=3  --generation_path="baselines/1101_subset/deis/cfg3/10step/" --type="deis" 
python gen_ppo.py --num_inference_steps 12 --cfg=3  --generation_path="baselines/1101_subset/deis/cfg3/12step/" --type="deis" 
python gen_ppo.py --num_inference_steps 15 --cfg=3  --generation_path="baselines/1101_subset/deis/cfg3/15step/" --type="deis" 


python gen_ppo.py --num_inference_steps 5 --cfg=3   --generation_path="baselines/1101_subset/amed/cfg3/5step/"  --type="amed"
python gen_ppo.py --num_inference_steps 8 --cfg=3   --generation_path="baselines/1101_subset/amed/cfg3/8step/"  --type="amed"
python gen_ppo.py --num_inference_steps 10 --cfg=3  --generation_path="baselines/1101_subset/amed/cfg3/10step/" --type="amed"
python gen_ppo.py --num_inference_steps 12 --cfg=3  --generation_path="baselines/1101_subset/amed/cfg3/12step/" --type="amed"
python gen_ppo.py --num_inference_steps 15 --cfg=3  --generation_path="baselines/1101_subset/amed/cfg3/15step/" --type="amed"


python gen_ppo.py --num_inference_steps 5 --cfg=3   --generation_path="baselines/1101_subset/ipndm/cfg3/5step/"  --type="ipndm"
python gen_ppo.py --num_inference_steps 8 --cfg=3   --generation_path="baselines/1101_subset/ipndm/cfg3/8step/"  --type="ipndm"
python gen_ppo.py --num_inference_steps 10 --cfg=3  --generation_path="baselines/1101_subset/ipndm/cfg3/10step/" --type="ipndm"
python gen_ppo.py --num_inference_steps 12 --cfg=3  --generation_path="baselines/1101_subset/ipndm/cfg3/12step/" --type="ipndm"
python gen_ppo.py --num_inference_steps 15 --cfg=3  --generation_path="baselines/1101_subset/ipndm/cfg3/15step/" --type="ipndm"


python gen_ppo.py --num_inference_steps 5 --cfg=3   --generation_path="baselines/1101_subset/unipc/cfg3/5step/"  --type="unipc"
python gen_ppo.py --num_inference_steps 8 --cfg=3   --generation_path="baselines/1101_subset/unipc/cfg3/8step/"  --type="unipc"
python gen_ppo.py --num_inference_steps 10 --cfg=3  --generation_path="baselines/1101_subset/unipc/cfg3/10step/" --type="unipc"
python gen_ppo.py --num_inference_steps 12 --cfg=3  --generation_path="baselines/1101_subset/unipc/cfg3/12step/" --type="unipc"
python gen_ppo.py --num_inference_steps 15 --cfg=3  --generation_path="baselines/1101_subset/unipc/cfg3/15step/" --type="unipc"