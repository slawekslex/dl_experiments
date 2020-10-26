import subprocess
import random



def draw_hypers():
    hypers=dict()
    hypers['max_tgt_length'] = random.randint(70,100)
    hypers['head_ps'] = random.uniform(0.5,.80)
    hypers['stem_ps'] = random.uniform(0.2,.30)
    hypers['head_linear'] = random.randint(400, 600)
    hypers['max_masked'] = random.randint(0, hypers['max_tgt_length'] //3)
    hypers['mask_prob'] = random.uniform(0, .30)
    hypers['vis_mask_prob'] = random.uniform(0.1, .30)
    hypers['lr'] = random.uniform(0.0002, 0.002)
    hypers['lr_mult'] = random.uniform(hypers['lr']/3e-5, hypers['lr']/1.5e-5)
    hypers['train_epochs'] = 8#random.randint(6, 12)
    hypers['stem_file'] = 'checkpoints/lm_stem_long.pth'
    return hypers
    
    


for exp_id in range(78, 120):
    print('============= exp', exp_id, '\n', flush=True)
    hypers = draw_hypers()    
    com = ['python', 'run_experiment.py', f'--experiment_id={exp_id}']
    for name,value in hypers.items():
        com.append(f'--{name}={value}')
    subprocess.run(com)