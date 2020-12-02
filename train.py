from pathlib import Path

from fastai.vision.widgets import *
from fastbook import *


def search_images_bing(key, term, max_images: int = 100, **kwargs):
    params = {'q':term, 'count':max_images}
    headers = {"Ocp-Apim-Subscription-Key":key}
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    return L(search_results['value'])

def prediction():
    path = Path()
    learn_inf = load_learner(path/'export.pkl')
    prediction = learn_inf.predict('test.jpg')
    return f"{prediction[0]} with probability {max(prediction[-1]):.4}"


poke_types = 'bulbasaur', 'ivysaur', 'venusaur', 'charmander', 'charmeleon', 'charizard', 'squirtle', 'wartortle', 'blastoise', 'caterpie', 'metapod', 'butterfree', 'weedle', 'kakuna', 'beedrill', 'pidgey', 'pidgeotto', 'pidgeot', 'rattata', 'raticate', 'spearow', 'fearow', 'ekans', 'arbok', 'pikachu', 'raichu', 'sandshrew', 'sandslash', 'nidoran-f', 'nidorina', 'nidoqueen', 'nidoran-m', 'nidorino', 'nidoking', 'clefairy', 'clefable', 'vulpix', 'ninetales', 'jigglypuff', 'wigglytuff', 'zubat', 'golbat', 'oddish', 'gloom', 'vileplume', 'paras', 'parasect', 'venonat', 'venomoth', 'diglett', 'dugtrio', 'meowth', 'persian', 'psyduck', 'golduck', 'mankey', 'primeape', 'growlithe', 'arcanine', 'poliwag', 'poliwhirl', 'poliwrath', 'abra', 'kadabra', 'alakazam', 'machop', 'machoke', 'machamp', 'bellsprout', 'weepinbell', 'victreebel', 'tentacool', 'tentacruel', 'geodude', 'graveler', 'golem', 'ponyta', 'rapidash', 'slowpoke', 'slowbro', 'magnemite', 'magneton', 'farfetchd', 'doduo', 'dodrio', 'seel', 'dewgong', 'grimer', 'muk', 'shellder', 'cloyster', 'gastly', 'haunter', 'gengar', 'onix', 'drowzee', 'hypno', 'krabby', 'kingler', 'voltorb', 'electrode', 'exeggcute', 'exeggutor', 'cubone', 'marowak', 'hitmonlee', 'hitmonchan', 'lickitung', 'koffing', 'weezing', 'rhyhorn', 'rhydon', 'chansey', 'tangela', 'kangaskhan', 'horsea', 'seadra', 'goldeen', 'seaking', 'staryu', 'starmie', 'mr-mime', 'scyther', 'jynx', 'electabuzz', 'magmar', 'pinsir', 'tauros', 'magikarp', 'gyarados', 'lapras', 'ditto', 'eevee', 'vaporeon', 'jolteon', 'flareon', 'porygon', 'omanyte', 'omastar', 'kabuto', 'kabutops', 'aerodactyl', 'snorlax', 'articuno', 'zapdos', 'moltres', 'dratini', 'dragonair', 'dragonite', 'mewtwo', 'mew'

setup_book()
key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')
path = Path('./pokemon')

c = 0
if path.exists():
    for o in poke_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o}')
        download_images(dest, urls=results.attrgot('contentUrl'), max_pics=1000)
        print(f"\r{c}{o}", end="")
        c+=1


fns = get_image_files(path)
print(fns)
failed = verify_images(fns)
print(failed)
failed.map(Path.unlink)

pokemon = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(256))

dls = pokemon.dataloaders(path)

dls.valid.show_batch(max_n=10, nrows=1)

pokemon = pokemon.new(item_tfms=RandomResizedCrop(128, min_scale=.2), batch_tfms=aug_transforms(mult=3))
dls = pokemon.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)

pokemon= pokemon.new(item_tfms=RandomResizedCrop(256, min_scale=0.2),batch_tfms=aug_transforms(mult=3))
dls = pokemon.dataloaders(path)

learn = cnn_learner(dls, resnet101 , metrics=error_rate)
learn.fine_tune(16)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(5, nrows=1)
cleaner = ImageClassifierCleaner(learn)

for idx in cleaner.delete():
    try:
        cleaner.fns[idx].unlink()
    except:
        pass
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)

learn.export(path/'export.pkl')
