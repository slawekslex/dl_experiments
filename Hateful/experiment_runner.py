import subprocess
import random



def draw_hypers():
    hypers=dict()
    hypers['max_tgt_length'] = random.randint(10,100)
    hypers['head_ps'] = random.uniform(0,.80)
    hypers['stem_ps'] = random.uniform(0,.50)
    hypers['head_linear'] = random.randint(50, 1024)
    hypers['max_masked'] = random.randint(0, hypers['max_tgt_length'])
    hypers['mask_prob'] = random.uniform(0, .30)
    hypers['vis_mask_prob'] = random.uniform(0, .30)
    hypers['lr'] = random.uniform(0.001 / 5, 0.001 * 5)
    hypers['lr_mult'] = random.randint(2, 50)
    hypers['train_epochs'] = random.randint(6, 12)
    return hypers
    
    


for exp_id in range(2, 40):
    print('============= exp', exp_id, '\n')
    hypers = draw_hypers()    
    com = ['python', 'run_experiment.py', f'--experiment_id={exp_id}']
    for name,value in hypers.items():
        com.append(f'--{name}={value}')
    subprocess.run(com)