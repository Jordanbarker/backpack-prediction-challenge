target = 'price'

baseline_features = ['weight_capacity', 'color', 'compartments', 'brand', 'material', 'is_waterproof']

cat_cols = ['brand', 'material', 'size', 'compartments', 'style', 'color', 'laptop_compartment', 'is_waterproof']
#, 'weight_capacity_int']


feature_list = [
    'material_weight_capacity_int_encoded',
    'brand_weight_capacity_int_encoded',
    'color_weight_capacity_int_encoded', 
    'size_weight_capacity_int_encoded',
    'is_waterproof_weight_capacity_int_encoded', 
    'weight_capacity',
    'material_color_encoded', 
    'compartments', 
    'size_compartments_encoded',
    'brand_size_encoded', 
    'compartments_weight_capacity_int_encoded',
    'brand_color_encoded', 
    'size_is_waterproof_encoded',
    'laptop_compartment_is_waterproof_encoded',
    'size_laptop_compartment_encoded', 
    'brand_material_encoded',
    'compartments_is_waterproof_encoded', 
    'size_color_encoded',
    'material_size_encoded', 
    'style_is_waterproof_encoded',
    'brand_style_encoded', 
    'style_laptop_compartment_encoded',
    'compartments_laptop_compartment_encoded', 
    'compartments_color_encoded',
    'material_compartments_encoded', 
    'brand_compartments_encoded',
    'material_is_waterproof_encoded', 
    'compartments_style_encoded',
    'style_weight_capacity_int_encoded',
    'material_laptop_compartment_encoded', 
    'color_is_waterproof_encoded',
# ]

# unused = [
    'material_style_encoded',
    'laptop_compartment_weight_capacity_int_encoded',
    'weight_capacity_size', 'style_color_encoded',
    'brand_laptop_compartment_encoded', 
    'color_laptop_compartment_encoded',
    'style_encoded', 
    'brand_is_waterproof_encoded', 
    'size_style_encoded',
    'weight_capacity_int_encoded', 
    'brand_encoded',
    'laptop_compartment_encoded',
    'compartments_encoded',
    'encoded_weight_capacity', 
    'color', 
    'material_encoded', 
    'color_encoded',
    'is_waterproof_encoded', 
    'size_encoded', 
    'weight_capacity_int',
    'laptop_compartment', 
    'brand', 
    'material', 
    'size_int', 
    'size', 
    'style',
    'is_waterproof', 
    'binned_weight_capacity'
]