
material_all_str = ["brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair", "leather",
              "metal", "mirror", "other", "painted", "paper", "plastic", "polishedstone",
              "skin", "sky", "stone", "tile", "wallpaper", "water", "wood"]

material_ipalm_ignore = ["mirror", "sky", "skin", "leather", "hair",
                        "painted", "brick", "carpet", "fabric",
                        "foliage", "food", "polishedstone", "stone", "tile",
                        "wallpaper"]

# [2, 6, 9, 11, 13, 14, 21, 22]
material_ipalm2raw = [i for i in range(len(material_all_str)) if material_all_str[i] not in material_ipalm_ignore]

# {2: 0, 6: 1, 9: 2, 11: 3, 13: 4, 14: 5, 21: 6, 22: 7}
material_raw2ipalm = {material_ipalm2raw[i]: i for i in range(len(material_ipalm2raw))}

# {2: 'ceramic', 6: 'glass', 9: 'metal', 11: 'other', 13: 'paper', 14: 'plastic', 21: 'water', 22: 'wood'}
# \material_ipalm2raw/
#  \=> {0: 'ceramic', 1: 'glass', 2: 'metal', 3: 'other', 4: 'paper', 5: 'plastic', 6: 'water', 7: 'wood'}
material_ipalm_id2str = {i: material_all_str[material_ipalm2raw[i]] for i in range(len(material_ipalm2raw))}

# {'ceramic': 0, 'glass': 1, 'metal': 2, 'other': 3, 'paper': 4, 'plastic': 5, 'water': 6, 'wood': 7}
material_str2ipalm_id = {material_all_str[material_ipalm2raw[i]]: i for i in range(len(material_ipalm2raw))}

# {'brick': 0, 'carpet': 1, 'ceramic': 2, 'fabric': 3, 'foliage': 4, 'food': 5, 'glass': 6, 'hair': 7, 'leather': 8,
#  'metal': 9, 'mirror': 10, 'other': 11, 'painted': 12, 'paper': 13, 'plastic': 14, 'polishedstone': 15, 'skin': 16,
#  'sky': 17, 'stone': 18, 'tile': 19, 'wallpaper': 20, 'water': 21, 'wood': 22}
material_str2raw_id = {material_all_str[i]: i for i in range(len(material_all_str))}
# print(material_str2raw_id)
# {0: 'brick', 1: 'carpet', 2: 'ceramic', 3: 'fabric', 4: 'foliage', 5: 'food', 6: 'glass', 7: 'hair', 8: 'leather',
#  9: 'metal', 10: 'mirror', 11: 'other', 12: 'painted', 13: 'paper', 14: 'plastic', 15: 'polishedstone', 16: 'skin',
#  17: 'sky', 18: 'stone', 19: 'tile', 20: 'wallpaper', 21: 'water', 22: 'wood'}
material_raw_id2str = {i: material_all_str[i] for i in range(len(material_all_str))}
# print(material_raw_id2str)

