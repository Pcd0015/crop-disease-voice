"""
Static disease reference library. Two tiers of entries live here:

1. PHOTO-DIAGNOSABLE (38 entries, is_diagnosable=True) — one entry per class
   the trained model can actually recognize (the PlantVillage classes, via
   the Kaggle abdallahalidev/plantvillage-dataset mirror used in
   training/train_disease_model.py). Keys match `class_names.json` exactly
   (Kaggle folder names), so these can be looked up directly with a
   DiagnosisResult.predicted_class value.

2. REFERENCE-ONLY (is_diagnosable=False) — additional major crops/diseases
   the model was NOT trained on and therefore CANNOT identify from a photo.
   These exist purely as a free browsable reference (Plantix-style), and
   their keys are prefixed "REF__" so they can never collide with or be
   mistaken for a real model prediction. The UI must always show these with
   a clear "reference only" label — never imply a photo diagnosis covers them.

General agronomic knowledge throughout, not sourced from any single
copyrighted text — kept short and factual. Not a substitute for a local
agricultural extension office or a lab test.
"""
from dataclasses import dataclass


@dataclass
class DiseaseInfo:
    crop: str
    condition: str          # "Healthy" or the disease/pest name
    is_healthy: bool
    symptoms: str
    prevention: str
    is_diagnosable: bool = True   # False = reference-only, not recognized by the photo model


DISEASE_LIBRARY: dict[str, DiseaseInfo] = {
    "Apple___Apple_scab": DiseaseInfo(
        "Apple", "Apple scab", False,
        "Olive-green to brown velvety spots on leaves and fruit; leaves may yellow and drop early.",
        "Rake and destroy fallen leaves each autumn, prune for airflow, and apply a protectant fungicide from bud break through wet spring weather.",
    ),
    "Apple___Black_rot": DiseaseInfo(
        "Apple", "Black rot", False,
        "Purple-bordered leaf spots and rotting, mummified fruit with concentric rings.",
        "Remove mummified fruit and cankered wood, prune dead branches, and keep the orchard floor clear of debris.",
    ),
    "Apple___Cedar_apple_rust": DiseaseInfo(
        "Apple", "Cedar apple rust", False,
        "Bright orange-yellow spots on leaves, sometimes with small raised dots.",
        "Remove nearby cedar/juniper hosts if possible, or apply protectant fungicide in spring when both hosts are present.",
    ),
    "Apple___healthy": DiseaseInfo(
        "Apple", "Healthy", True,
        "No visible disease symptoms — leaves are uniformly green with no spots, wilting, or discoloration.",
        "Maintain regular watering, balanced fertilization, and annual pruning for airflow.",
    ),
    "Blueberry___healthy": DiseaseInfo(
        "Blueberry", "Healthy", True,
        "No visible disease symptoms.",
        "Keep soil acidic (pH 4.5-5.5), mulch to retain moisture, and prune for airflow.",
    ),
    "Cherry_(including_sour)___Powdery_mildew": DiseaseInfo(
        "Cherry", "Powdery mildew", False,
        "White powdery coating on leaves and shoots, often curling young leaves.",
        "Prune for airflow, avoid overhead watering, and apply sulfur-based fungicide early if humidity is high.",
    ),
    "Cherry_(including_sour)___healthy": DiseaseInfo(
        "Cherry", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good airflow through pruning and avoid overhead irrigation.",
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": DiseaseInfo(
        "Corn (maize)", "Gray leaf spot", False,
        "Rectangular tan-to-gray lesions running parallel to leaf veins.",
        "Rotate crops away from corn for a season, choose resistant varieties, and manage crop residue.",
    ),
    "Corn_(maize)___Common_rust_": DiseaseInfo(
        "Corn (maize)", "Common rust", False,
        "Small, reddish-brown raised pustules scattered on both leaf surfaces.",
        "Plant resistant hybrids; fungicide is rarely needed unless pressure is severe early in the season.",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": DiseaseInfo(
        "Corn (maize)", "Northern leaf blight", False,
        "Long, gray-green, cigar-shaped lesions on lower leaves that spread upward.",
        "Rotate crops, till under residue, and choose resistant hybrids where available.",
    ),
    "Corn_(maize)___healthy": DiseaseInfo(
        "Corn (maize)", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain balanced nitrogen and adequate spacing for airflow.",
    ),
    "Grape___Black_rot": DiseaseInfo(
        "Grape", "Black rot", False,
        "Small tan spots with dark borders on leaves; fruit shrivels into hard black mummies.",
        "Remove mummified fruit and infected canes, and apply fungicide starting at bud break in wet climates.",
    ),
    "Grape___Esca_(Black_Measles)": DiseaseInfo(
        "Grape", "Esca (Black Measles)", False,
        "Tiger-stripe pattern of yellow/red discoloration between leaf veins; fruit may show dark spots.",
        "Prune out and destroy infected wood; avoid pruning wounds during wet weather.",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": DiseaseInfo(
        "Grape", "Leaf blight (Isariopsis leaf spot)", False,
        "Angular reddish-brown spots on leaves that can merge and cause early leaf drop.",
        "Improve canopy airflow through pruning and apply fungicide during humid periods.",
    ),
    "Grape___healthy": DiseaseInfo(
        "Grape", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain canopy airflow through pruning and avoid excess nitrogen.",
    ),
    "Orange___Haunglongbing_(Citrus_greening)": DiseaseInfo(
        "Orange (Citrus)", "Citrus greening (Huanglongbing)", False,
        "Blotchy yellow mottling on leaves (asymmetric across the midrib), small lopsided bitter fruit.",
        "No cure exists — remove and destroy infected trees, and control the Asian citrus psyllid insect that spreads it. Contact your local agricultural extension office if suspected.",
    ),
    "Peach___Bacterial_spot": DiseaseInfo(
        "Peach", "Bacterial spot", False,
        "Small dark, water-soaked spots on leaves and fruit that may fall out, leaving a shot-hole look.",
        "Plant resistant varieties, avoid overhead irrigation, and apply copper-based sprays before symptoms appear in wet seasons.",
    ),
    "Peach___healthy": DiseaseInfo(
        "Peach", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain regular pruning and balanced fertilization.",
    ),
    "Pepper,_bell___Bacterial_spot": DiseaseInfo(
        "Bell pepper", "Bacterial spot", False,
        "Small water-soaked spots on leaves that turn brown/scabby; fruit may show raised scabby spots.",
        "Use disease-free seed, avoid overhead watering, and rotate crops away from peppers/tomatoes for at least a year.",
    ),
    "Pepper,_bell___healthy": DiseaseInfo(
        "Bell pepper", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain even watering and balanced fertilization.",
    ),
    "Potato___Early_blight": DiseaseInfo(
        "Potato", "Early blight", False,
        "Dark brown spots with concentric rings (target-like pattern), usually starting on older leaves.",
        "Rotate crops, remove infected debris, and apply fungicide preventively in warm, humid conditions.",
    ),
    "Potato___Late_blight": DiseaseInfo(
        "Potato", "Late blight", False,
        "Water-soaked gray-green patches that turn brown/black quickly, often with white mold on leaf undersides in humid weather.",
        "Destroy infected plants promptly, avoid overhead watering, and apply fungicide preventively when humidity is high — this disease spreads fast.",
    ),
    "Potato___healthy": DiseaseInfo(
        "Potato", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain hilled rows, even watering, and crop rotation.",
    ),
    "Raspberry___healthy": DiseaseInfo(
        "Raspberry", "Healthy", True,
        "No visible disease symptoms.",
        "Prune out old canes after fruiting and maintain airflow.",
    ),
    "Soybean___healthy": DiseaseInfo(
        "Soybean", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and maintain balanced soil fertility.",
    ),
    "Squash___Powdery_mildew": DiseaseInfo(
        "Squash", "Powdery mildew", False,
        "White powdery patches on leaf surfaces, usually starting on older leaves.",
        "Plant resistant varieties, ensure good spacing for airflow, and apply sulfur or potassium bicarbonate spray early.",
    ),
    "Strawberry___Leaf_scorch": DiseaseInfo(
        "Strawberry", "Leaf scorch", False,
        "Small purple spots that merge into larger scorched-looking brown patches.",
        "Remove infected leaves after harvest, avoid overhead watering, and ensure good airflow between plants.",
    ),
    "Strawberry___healthy": DiseaseInfo(
        "Strawberry", "Healthy", True,
        "No visible disease symptoms.",
        "Mulch to keep fruit off soil and maintain even moisture.",
    ),
    "Tomato___Bacterial_spot": DiseaseInfo(
        "Tomato", "Bacterial spot", False,
        "Small dark, greasy-looking spots on leaves and fruit, sometimes with a yellow halo.",
        "Use disease-free seed/transplants, avoid overhead watering, and rotate crops away from tomatoes/peppers.",
    ),
    "Tomato___Early_blight": DiseaseInfo(
        "Tomato", "Early blight", False,
        "Dark spots with concentric target-like rings, typically starting on lower/older leaves.",
        "Remove lower infected leaves, mulch to prevent soil splash, and rotate crops each season.",
    ),
    "Tomato___Late_blight": DiseaseInfo(
        "Tomato", "Late blight", False,
        "Large water-soaked gray-green patches that turn brown quickly, often with white fuzzy mold underneath in humid weather.",
        "Act quickly — remove infected plants, avoid overhead watering, and apply fungicide preventively in humid conditions. Spreads fast.",
    ),
    "Tomato___Leaf_Mold": DiseaseInfo(
        "Tomato", "Leaf mold", False,
        "Pale yellow spots on upper leaf surface with olive-green to gray fuzzy mold underneath.",
        "Improve greenhouse/field ventilation, avoid overhead watering, and space plants for airflow.",
    ),
    "Tomato___Septoria_leaf_spot": DiseaseInfo(
        "Tomato", "Septoria leaf spot", False,
        "Small circular spots with dark borders and tan/gray centers, usually starting on lower leaves.",
        "Remove infected lower leaves, mulch to prevent soil splash, and rotate crops.",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": DiseaseInfo(
        "Tomato", "Two-spotted spider mite", False,
        "Fine yellow stippling on leaves, sometimes fine webbing on the underside in heavy infestations.",
        "Spray leaves with water to dislodge mites, encourage natural predators, and use insecticidal soap if severe. Avoid drought stress on plants.",
    ),
    "Tomato___Target_Spot": DiseaseInfo(
        "Tomato", "Target spot", False,
        "Brown lesions with concentric rings, similar to early blight, on leaves and sometimes fruit.",
        "Remove infected debris, improve airflow through pruning/spacing, and rotate crops.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DiseaseInfo(
        "Tomato", "Tomato yellow leaf curl virus", False,
        "Upward-curling, yellowing leaves and stunted plant growth; spread by whiteflies.",
        "Control whitefly populations, remove and destroy infected plants promptly, and use resistant varieties where available.",
    ),
    "Tomato___Tomato_mosaic_virus": DiseaseInfo(
        "Tomato", "Tomato mosaic virus", False,
        "Mottled light/dark green mosaic pattern on leaves, sometimes with leaf distortion.",
        "Remove and destroy infected plants, wash hands/tools between plants (the virus spreads by contact), and avoid tobacco product contact with plants.",
    ),
    "Tomato___healthy": DiseaseInfo(
        "Tomato", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain even watering, staking for airflow, and crop rotation each season.",
    ),
}


# ---------------------------------------------------------------------------
# REFERENCE-ONLY entries — 26 additional major crops, not recognized by the
# photo model. Browsable reference information only. Keys are prefixed
# "REF__" so they can never be returned by / confused with a real diagnosis.
# ---------------------------------------------------------------------------
REFERENCE_LIBRARY: dict[str, DiseaseInfo] = {
    "REF__Rice___Blast": DiseaseInfo(
        "Rice", "Rice blast", False,
        "Diamond-shaped lesions with gray centers and brown margins on leaves; can also strike the neck of the panicle.",
        "Use resistant varieties, avoid excess nitrogen, and keep fields from drying out and re-flooding repeatedly.",
        is_diagnosable=False,
    ),
    "REF__Rice___Bacterial_leaf_blight": DiseaseInfo(
        "Rice", "Bacterial leaf blight", False,
        "Water-soaked streaks starting at leaf tips or margins that turn yellow-white and dry out.",
        "Use disease-free seed, avoid excess nitrogen, and drain standing water where the disease has appeared before.",
        is_diagnosable=False,
    ),
    "REF__Rice___healthy": DiseaseInfo(
        "Rice", "Healthy", True,
        "No visible disease symptoms — uniform green leaves with no lesions or streaking.",
        "Maintain balanced fertilization and proper water management through the season.",
        is_diagnosable=False,
    ),
    "REF__Wheat___Yellow_rust": DiseaseInfo(
        "Wheat", "Yellow (stripe) rust", False,
        "Yellow-orange powdery pustules arranged in stripes along the leaf veins.",
        "Plant resistant varieties, sow on time to avoid peak disease periods, and monitor closely in cool, humid weather.",
        is_diagnosable=False,
    ),
    "REF__Wheat___Powdery_mildew": DiseaseInfo(
        "Wheat", "Powdery mildew", False,
        "White powdery patches on leaves and stems, most common in cool, humid conditions with dense canopy.",
        "Avoid overly dense sowing, avoid excess nitrogen, and choose resistant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Wheat___healthy": DiseaseInfo(
        "Wheat", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain timely sowing, balanced fertilization, and field monitoring through the growing season.",
        is_diagnosable=False,
    ),
    "REF__Cotton___Bacterial_blight": DiseaseInfo(
        "Cotton", "Bacterial blight (angular leaf spot)", False,
        "Angular, water-soaked lesions on leaves that darken and can spread to stems and bolls.",
        "Use certified disease-free seed, rotate crops, and remove and destroy infected plant debris after harvest.",
        is_diagnosable=False,
    ),
    "REF__Cotton___Leaf_curl_virus": DiseaseInfo(
        "Cotton", "Cotton leaf curl virus", False,
        "Upward curling and thickening of leaves with stunted plant growth; spread by whitefly.",
        "Control whitefly populations, remove infected plants promptly, and use tolerant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Cotton___healthy": DiseaseInfo(
        "Cotton", "Healthy", True,
        "No visible disease symptoms.",
        "Monitor for whitefly and maintain balanced irrigation and fertilization.",
        is_diagnosable=False,
    ),
    "REF__Sugarcane___Red_rot": DiseaseInfo(
        "Sugarcane", "Red rot", False,
        "Reddish discoloration inside the stalk with white patches, plus wilting and drying of leaves.",
        "Plant disease-free setts from a healthy source, rotate out of sugarcane for a season, and avoid waterlogging.",
        is_diagnosable=False,
    ),
    "REF__Sugarcane___Smut": DiseaseInfo(
        "Sugarcane", "Smut", False,
        "A long, black, whip-like structure emerging from the growing point of the plant.",
        "Remove and destroy whips before they rupture and release spores, and plant resistant varieties.",
        is_diagnosable=False,
    ),
    "REF__Sugarcane___healthy": DiseaseInfo(
        "Sugarcane", "Healthy", True,
        "No visible disease symptoms.",
        "Use disease-free planting material and maintain field sanitation.",
        is_diagnosable=False,
    ),
    "REF__Banana___Panama_wilt": DiseaseInfo(
        "Banana", "Panama wilt (Fusarium wilt)", False,
        "Yellowing of older leaves that progresses upward, splitting at the base of the pseudostem, and eventual collapse.",
        "Plant resistant varieties, avoid moving soil/water from infected fields, and remove and destroy infected plants — there is no cure once infected.",
        is_diagnosable=False,
    ),
    "REF__Banana___Black_Sigatoka": DiseaseInfo(
        "Banana", "Black Sigatoka", False,
        "Dark brown-black streaks on leaves that merge into dead patches, reducing photosynthesis and fruit yield.",
        "Remove and destroy heavily infected leaves, improve field spacing for airflow, and use resistant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Banana___healthy": DiseaseInfo(
        "Banana", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good drainage and remove old, damaged leaves regularly.",
        is_diagnosable=False,
    ),
    "REF__Mango___Anthracnose": DiseaseInfo(
        "Mango", "Anthracnose", False,
        "Black, sunken spots on leaves, flowers, and fruit; can cause flower and young fruit drop.",
        "Prune for airflow, remove fallen debris, and apply protectant fungicide during flowering in humid weather.",
        is_diagnosable=False,
    ),
    "REF__Mango___Powdery_mildew": DiseaseInfo(
        "Mango", "Powdery mildew", False,
        "White powdery growth on young leaves, flowers, and fruit, often causing flower and fruit drop.",
        "Apply sulfur-based fungicide at flowering onset in cool, humid conditions, and prune for airflow.",
        is_diagnosable=False,
    ),
    "REF__Mango___healthy": DiseaseInfo(
        "Mango", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain regular pruning and monitor closely during flowering.",
        is_diagnosable=False,
    ),
    "REF__Chili___Anthracnose": DiseaseInfo(
        "Chili (Pepper)", "Anthracnose (fruit rot)", False,
        "Sunken, circular dark spots on ripening fruit, often with concentric rings.",
        "Use disease-free seed, avoid overhead watering, and rotate crops away from chili/tomato for at least a year.",
        is_diagnosable=False,
    ),
    "REF__Chili___Leaf_curl_virus": DiseaseInfo(
        "Chili (Pepper)", "Chili leaf curl virus", False,
        "Upward curling, crinkling of leaves and stunted plant growth; spread by whitefly.",
        "Control whitefly populations, remove infected plants promptly, and use tolerant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Chili___healthy": DiseaseInfo(
        "Chili (Pepper)", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain even watering and monitor for whitefly.",
        is_diagnosable=False,
    ),
    "REF__Cucumber___Downy_mildew": DiseaseInfo(
        "Cucumber", "Downy mildew", False,
        "Yellow, angular spots on the upper leaf surface with grayish-purple mold underneath.",
        "Improve airflow through spacing, avoid overhead watering, and apply protectant fungicide in humid weather.",
        is_diagnosable=False,
    ),
    "REF__Cucumber___Powdery_mildew": DiseaseInfo(
        "Cucumber", "Powdery mildew", False,
        "White, powdery coating on leaves and stems.",
        "Plant resistant varieties, ensure good spacing, and apply sulfur or potassium bicarbonate spray early.",
        is_diagnosable=False,
    ),
    "REF__Cucumber___healthy": DiseaseInfo(
        "Cucumber", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good airflow and avoid overhead irrigation.",
        is_diagnosable=False,
    ),
    "REF__Onion___Purple_blotch": DiseaseInfo(
        "Onion", "Purple blotch", False,
        "Purple-brown lesions with concentric zones on leaves, often starting at older leaf tips.",
        "Avoid overhead watering, rotate crops, and apply protectant fungicide during humid periods.",
        is_diagnosable=False,
    ),
    "REF__Onion___healthy": DiseaseInfo(
        "Onion", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good drainage and avoid overcrowding.",
        is_diagnosable=False,
    ),
    "REF__Cabbage___Black_rot": DiseaseInfo(
        "Cabbage", "Black rot", False,
        "V-shaped yellow lesions starting at leaf margins, with blackened veins.",
        "Use disease-free seed, rotate crops for at least two years, and avoid working in wet fields.",
        is_diagnosable=False,
    ),
    "REF__Cabbage___healthy": DiseaseInfo(
        "Cabbage", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and maintain balanced soil fertility.",
        is_diagnosable=False,
    ),
    "REF__Cauliflower___Downy_mildew": DiseaseInfo(
        "Cauliflower", "Downy mildew", False,
        "Yellow patches on the upper leaf surface with gray-white mold underneath, common in cool, damp weather.",
        "Improve airflow, avoid overhead watering, and rotate crops.",
        is_diagnosable=False,
    ),
    "REF__Cauliflower___healthy": DiseaseInfo(
        "Cauliflower", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain even moisture and good field drainage.",
        is_diagnosable=False,
    ),
    "REF__Groundnut___Tikka_leaf_spot": DiseaseInfo(
        "Groundnut (Peanut)", "Tikka leaf spot", False,
        "Small, circular brown-black spots on leaves with a yellow halo, causing early leaf drop.",
        "Rotate crops, remove infected debris after harvest, and use resistant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Groundnut___healthy": DiseaseInfo(
        "Groundnut (Peanut)", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and maintain balanced soil fertility.",
        is_diagnosable=False,
    ),
    "REF__Chickpea___Ascochyta_blight": DiseaseInfo(
        "Chickpea (Gram)", "Ascochyta blight", False,
        "Brown lesions with concentric rings on leaves, stems, and pods.",
        "Use disease-free seed, rotate crops, and avoid working fields when foliage is wet.",
        is_diagnosable=False,
    ),
    "REF__Chickpea___healthy": DiseaseInfo(
        "Chickpea (Gram)", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and use certified seed.",
        is_diagnosable=False,
    ),
    "REF__Mustard___White_rust": DiseaseInfo(
        "Mustard", "White rust", False,
        "White, raised pustules on the underside of leaves, sometimes distorting stems and flowers.",
        "Rotate crops, remove volunteer plants and infected debris, and use resistant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Mustard___healthy": DiseaseInfo(
        "Mustard", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain balanced fertilization and timely sowing.",
        is_diagnosable=False,
    ),
    "REF__Sunflower___Downy_mildew": DiseaseInfo(
        "Sunflower", "Downy mildew", False,
        "Pale green-yellow patches on the upper leaf surface, stunted plants, and white cottony growth underneath in humid conditions.",
        "Use treated, disease-free seed, rotate crops, and improve field drainage.",
        is_diagnosable=False,
    ),
    "REF__Sunflower___healthy": DiseaseInfo(
        "Sunflower", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and avoid waterlogged soil.",
        is_diagnosable=False,
    ),
    "REF__Brinjal___Bacterial_wilt": DiseaseInfo(
        "Brinjal (Eggplant)", "Bacterial wilt", False,
        "Sudden wilting of the whole plant despite moist soil, without yellowing first.",
        "Rotate crops away from solanaceous vegetables, improve drainage, and remove and destroy infected plants promptly.",
        is_diagnosable=False,
    ),
    "REF__Brinjal___healthy": DiseaseInfo(
        "Brinjal (Eggplant)", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and avoid waterlogging.",
        is_diagnosable=False,
    ),
    "REF__Watermelon___Fusarium_wilt": DiseaseInfo(
        "Watermelon", "Fusarium wilt", False,
        "Wilting of vines starting with older leaves, often one-sided, with brown streaking inside the stem.",
        "Use resistant varieties, rotate crops for several years, and avoid replanting in previously infected soil.",
        is_diagnosable=False,
    ),
    "REF__Watermelon___healthy": DiseaseInfo(
        "Watermelon", "Healthy", True,
        "No visible disease symptoms.",
        "Rotate crops and maintain even soil moisture.",
        is_diagnosable=False,
    ),
    "REF__Papaya___Ring_spot_virus": DiseaseInfo(
        "Papaya", "Ring spot virus", False,
        "Mottled yellow leaves with distinctive ring-shaped spots on fruit, and distorted new growth.",
        "Control aphid vectors, remove and destroy infected plants promptly, and use tolerant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Papaya___healthy": DiseaseInfo(
        "Papaya", "Healthy", True,
        "No visible disease symptoms.",
        "Monitor for aphids and maintain good field sanitation.",
        is_diagnosable=False,
    ),
    "REF__Guava___Wilt_disease": DiseaseInfo(
        "Guava", "Wilt disease", False,
        "Gradual yellowing and wilting of branches, with dieback typically starting from twig tips.",
        "Remove and destroy severely affected trees, improve soil drainage, and avoid injuring roots during cultivation.",
        is_diagnosable=False,
    ),
    "REF__Guava___healthy": DiseaseInfo(
        "Guava", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good drainage and regular pruning.",
        is_diagnosable=False,
    ),
    "REF__Pomegranate___Bacterial_blight": DiseaseInfo(
        "Pomegranate", "Bacterial blight", False,
        "Dark, irregular spots on leaves and fruit with cracking, plus oily-looking lesions on twigs.",
        "Prune out infected twigs, avoid overhead irrigation, and disinfect pruning tools between plants.",
        is_diagnosable=False,
    ),
    "REF__Pomegranate___healthy": DiseaseInfo(
        "Pomegranate", "Healthy", True,
        "No visible disease symptoms.",
        "Prune for airflow and avoid overhead watering.",
        is_diagnosable=False,
    ),
    "REF__Turmeric___Leaf_spot": DiseaseInfo(
        "Turmeric", "Leaf spot", False,
        "Small, oval brown spots with yellow margins on leaves, reducing rhizome yield if severe.",
        "Avoid overcrowding for better airflow, remove infected leaves, and rotate crops.",
        is_diagnosable=False,
    ),
    "REF__Turmeric___healthy": DiseaseInfo(
        "Turmeric", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good drainage and balanced fertilization.",
        is_diagnosable=False,
    ),
    "REF__Ginger___Rhizome_rot": DiseaseInfo(
        "Ginger", "Rhizome rot (soft rot)", False,
        "Yellowing and wilting of shoots, with water-soaked soft rot at the rhizome base and a foul smell.",
        "Plant disease-free rhizomes in well-drained soil, and avoid waterlogging — the main driver of this disease.",
        is_diagnosable=False,
    ),
    "REF__Ginger___healthy": DiseaseInfo(
        "Ginger", "Healthy", True,
        "No visible disease symptoms.",
        "Ensure good drainage and use disease-free planting material.",
        is_diagnosable=False,
    ),
    "REF__Garlic___Purple_blotch": DiseaseInfo(
        "Garlic", "Purple blotch", False,
        "Purple-brown lesions with concentric rings on leaves, similar in appearance to the same disease in onion.",
        "Avoid overhead watering, rotate crops, and apply protectant fungicide during humid periods.",
        is_diagnosable=False,
    ),
    "REF__Garlic___healthy": DiseaseInfo(
        "Garlic", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain good drainage and avoid overcrowding.",
        is_diagnosable=False,
    ),
    "REF__Okra___Yellow_vein_mosaic_virus": DiseaseInfo(
        "Okra (Lady's finger)", "Yellow vein mosaic virus", False,
        "A bright yellow network of veins on leaves, with pale and stunted fruit; spread by whitefly.",
        "Control whitefly populations, remove infected plants promptly, and use tolerant varieties where available.",
        is_diagnosable=False,
    ),
    "REF__Okra___healthy": DiseaseInfo(
        "Okra (Lady's finger)", "Healthy", True,
        "No visible disease symptoms.",
        "Monitor for whitefly and maintain even watering.",
        is_diagnosable=False,
    ),
    "REF__Carrot___Leaf_blight": DiseaseInfo(
        "Carrot", "Leaf blight (Alternaria)", False,
        "Dark brown-black spots on leaf margins and tips, causing leaves to wither.",
        "Use disease-free seed, avoid overhead watering, and rotate crops.",
        is_diagnosable=False,
    ),
    "REF__Carrot___healthy": DiseaseInfo(
        "Carrot", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain loose, well-drained soil and rotate crops.",
        is_diagnosable=False,
    ),
    "REF__Spinach___Downy_mildew": DiseaseInfo(
        "Spinach", "Downy mildew", False,
        "Pale yellow patches on the upper leaf surface with gray-purple fuzzy mold underneath.",
        "Improve airflow through spacing, avoid overhead watering, and rotate crops.",
        is_diagnosable=False,
    ),
    "REF__Spinach___healthy": DiseaseInfo(
        "Spinach", "Healthy", True,
        "No visible disease symptoms.",
        "Maintain even moisture and good field drainage.",
        is_diagnosable=False,
    ),
}

DISEASE_LIBRARY.update(REFERENCE_LIBRARY)


def get_all_crops() -> list[str]:
    return sorted({info.crop for info in DISEASE_LIBRARY.values()})


def get_entries_for_crop(crop: str) -> list[tuple[str, DiseaseInfo]]:
    return sorted(
        [(k, v) for k, v in DISEASE_LIBRARY.items() if v.crop == crop],
        key=lambda kv: kv[1].condition,
    )


def count_diagnosable() -> int:
    return sum(1 for info in DISEASE_LIBRARY.values() if info.is_diagnosable)


def count_reference_only() -> int:
    return sum(1 for info in DISEASE_LIBRARY.values() if not info.is_diagnosable)


def get_diagnosable_crops() -> list[str]:
    return sorted({info.crop for info in DISEASE_LIBRARY.values() if info.is_diagnosable})
