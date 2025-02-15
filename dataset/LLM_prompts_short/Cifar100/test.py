import json

prompt_dir = f'./text_encoder_prompts.json'
with open(prompt_dir, 'r') as f:
    text_encoder_prompts_new = json.load(f)


classname= ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"]

print(len(text_encoder_prompts_new))

for name in classname:
    print(text_encoder_prompts_new[name]['Fine-grained Descriptions'])


# for name in classname:
#     old_prompt = text_encoder_prompts_old[name]["Fine-grained Descriptions"]
#     new_prompt = text_encoder_prompts_new[name]["Fine-grained Descriptions"]
#     prompt = old_prompt + new_prompt
#     text_encoder_prompts_old[name]["Fine-grained Descriptions"] = prompt
#
# with open(f'./text_encoder_prompts.json', 'w') as f:
#     json.dump(text_encoder_prompts_old, f, indent=4)

# number = 3
# class_prompts = [f"A photo of a {classname[0]}: {desc}" for desc in text_encoder_prompts_new[classname[0]]["Fine-grained Descriptions"][0:number]]
# print(class_prompts)
# print(type(text_encoder_prompts_new[classname[0]]["Fine-grained Descriptions"]))