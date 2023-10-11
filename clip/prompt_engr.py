import os
import json
import numpy as np
import json
import torch

OBJ = [
    'airplane',
    'apple',
    'backpack',
    'banana',
    'baseball_bat',
    'baseball_glove',
    'bear',
    'bed',
    'bench',
    'bicycle',
    'bird',
    'boat',
    'book',
    'bottle',
    'bowl',
    'broccoli',
    'bus',
    'cake',
    'car',
    'carrot',
    'cat',
    'cell_phone',
    'chair',
    'clock',
    'couch',
    'cow',
    'cup',
    'dining_table',
    'dog',
    'donut',
    'elephant',
    'fire_hydrant',
    'fork',
    'frisbee',
    'giraffe',
    'hair_drier',
    'handbag',
    'horse',
    'hot_dog',
    'keyboard',
    'kite',
    'knife',
    'laptop',
    'microwave',
    'motorcycle',
    'mouse',
    'orange',
    'oven',
    'parking_meter',
    'person',
    'pizza',
    'potted_plant',
    'refrigerator',
    'remote',
    'sandwich',
    'scissors',
    'sheep',
    'sink',
    'skateboard',
    'skis',
    'snowboard',
    'spoon',
    'sports_ball',
    'stop_sign',
    'suitcase',
    'surfboard',
    'teddy_bear',
    'tennis_racket',
    'tie',
    'toaster',
    'toilet',
    'toothbrush',
    'traffic_light',
    'train',
    'truck',
    'tv',
    'umbrella',
    'vase',
    'wine_glass',
    'zebra'
]

OBJ_TO_IND = {}
for i, o in enumerate(OBJ):
    OBJ_TO_IND[o] = i

VERB = [
    'adjust',
    'assemble',
    'block',
    'blow',
    'board',
    'break',
    'brush_with',
    'buy',
    'carry',
    'catch',
    'chase',
    'check',
    'clean',
    'control',
    'cook',
    'cut',
    'cut_with',
    'direct',
    'drag',
    'dribble',
    'drink_with',
    'drive',
    'dry',
    'eat',
    'eat_at',
    'exit',
    'feed',
    'fill',
    'flip',
    'flush',
    'fly',
    'greet',
    'grind',
    'groom',
    'herd',
    'hit',
    'hold',
    'hop_on',
    'hose',
    'hug',
    'hunt',
    'inspect',
    'install',
    'jump',
    'kick',
    'kiss',
    'lasso',
    'launch',
    'lick',
    'lie_on',
    'lift',
    'light',
    'load',
    'lose',
    'make',
    'milk',
    'move',
    'no_interaction',
    'open',
    'operate',
    'pack',
    'paint',
    'park',
    'pay',
    'peel',
    'pet',
    'pick',
    'pick_up',
    'point',
    'pour',
    'pull',
    'push',
    'race',
    'read',
    'release',
    'repair',
    'ride',
    'row',
    'run',
    'sail',
    'scratch',
    'serve',
    'set',
    'shear',
    'sign',
    'sip',
    'sit_at',
    'sit_on',
    'slide',
    'smell',
    'spin',
    'squeeze',
    'stab',
    'stand_on',
    'stand_under',
    'stick',
    'stir',
    'stop_at',
    'straddle',
    'swing',
    'tag',
    'talk_on',
    'teach',
    'text_on',
    'throw',
    'tie',
    'toast',
    'train',
    'turn',
    'type_on',
    'walk',
    'wash',
    'watch',
    'wave',
    'wear',
    'wield',
    'zip',
]

VERB_TO_IND = {}
for i, v in enumerate(VERB):
    VERB_TO_IND[v] = i

class PromptGenerator:
    def __init__(self, data_dir, map_func):
        self.prompts = []
        self.prompt_to_hoi = []
        self.hoi_to_prompt = []

        with open(os.path.join(data_dir, 'anno.json')) as f:
            j = json.load(f)
            for i, hoi in enumerate(j['hoi_list']):
                verb = hoi['verb']
                obj = hoi['obj']
                sym = hoi['sym']

                prompt = map_func(i, verb, sym, obj)
                self.prompts.extend(prompt)
                self.hoi_to_prompt.append(np.arange(len(self.prompt_to_hoi), len(self.prompt_to_hoi)+len(prompt)))
                self.prompt_to_hoi.extend([i] * len(prompt))

        self.prompt_to_hoi = np.array(self.prompt_to_hoi)
        print('Total prompts: ', len(self.prompts))
    
    def prompts_to_json(self, out_fp):
        prompts = np.array(self.prompts)
        j = {}
        for i in range(600):
            j[i] = prompts[self.hoi_to_prompt[i]].tolist()
        
        with open(out_fp, 'w') as f:
            json.dump(j, f)
    
    def prompts_to_json_list(self, out_fp):
        prompts = np.array(self.prompts)
        j = []
        for i in range(600):
            j.append(prompts[self.hoi_to_prompt[i]][0])
        
        with open(out_fp, 'w') as f:
            json.dump(j, f)

    def map_prompt_sigmoid_to_hoi(self, prompts_logits, collate='max'):
        assert prompts_logits.shape[1] == len(self.prompts)
        assert collate in ['max', 'mean']

        N, K = prompts_logits.shape
        hoi_logits = torch.zeros([N, 600], device=prompts_logits.device)

        for i in range(600):
            if collate == 'max':
                hoi_logits[:, i] = torch.max(prompts_logits[:, self.hoi_to_prompt[i]], dim=-1)[0]
            else:
                hoi_logits[:, i] = torch.mean(prompts_logits[:, self.hoi_to_prompt[i]], dim=-1)
                
        sigmoids = torch.sigmoid(hoi_logits / 10)
        softmaxs = torch.softmax(hoi_logits, dim=1)

        return sigmoids, softmaxs

def prompt1(i, verb, syms, obj):
    prompts = []
    if verb == 'no_interaction':
        prompts.append('a photo of a person and a %s' % obj)
    else:
        prompts = prompt1_json[str(i)]
        prompts = ['a photo of a person ' + x for x in prompts]
    return prompts


if __name__ == '__main__':
    with open('/mnt/4t/hico/prompts.json') as f:
        global prompt1_json
        prompt1_json = json.load(f)

    prompt = PromptGenerator('/mnt/4t/hico', prompt1)
    prompt.prompts_to_json_list('/mnt/4t/hico/prompts.json')
    