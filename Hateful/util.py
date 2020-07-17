import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import PIL

HATE_IMAGES = Path('/home/jupyter/mmf_data/datasets/hateful_memes/defaults/images')

def gen_submit(learn, path, fname):
    test_df = pd.read_json(path/'test.jsonl', lines=True)
    test_dl = learn.dls.test_dl(test_df)
    preds = learn.get_preds(dl=test_dl)[0]
    _, indcs = preds.max(dim=1)
    probs= preds[:,1]
    submit_df = pd.DataFrame()
    submit_df['id'] = test_df['id']
    submit_df['proba'] = probs
    submit_df['label'] = indcs
    submit_df = submit_df.set_index('id')
    submit_df.to_csv(fname, header=True)
    
def find_like(s, data):
    return data[data.text.str.contains(s, case=False)]

def by_ids(ids, data):
    return data[data.id.isin(ids)]

def show(data):
    n = min(len(data), 40)
    _,axs = plt.subplots((n+1)//2,2, figsize=(20,2*n))
    for ax, (_,row) in zip(axs.flatten(), data.iterrows()):
        img_path = HATE_IMAGES / row['img']
        ax.imshow(PIL.Image.open(img_path))
        ax.axis('off')
        clr = 'red' if row['label']==1 else 'green'
        txt = f'{row["id"]}: {row["tex_cap"][:80]}'
        ax.set_title(txt, color=clr) 