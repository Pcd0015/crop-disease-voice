"""
Static disease reference library — one entry per class the trained model
can recognize (the 38 PlantVillage classes, via the Kaggle
abdallahalidev/plantvillage-dataset mirror used in training/train_disease_model.py).

This is a free, no-API-cost feature modeled on Plantix's disease library:
farmers can browse symptoms/prevention for any of the 38 classes without
uploading a photo. General agronomic knowledge, not sourced from any single
copyrighted text — kept short and factual.

Keys match `class_names.json` exactly (Kaggle folder names), so this can be
looked up directly with a DiagnosisResult.predicted_class value.
"""
from dataclasses import dataclass


@dataclass
class DiseaseInfo:
    crop: str
    condition: str          # "Healthy" or the disease/pest name
    is_healthy: bool
    symptoms: str
    prevention: str


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


def get_all_crops() -> list[str]:
    return sorted({info.crop for info in DISEASE_LIBRARY.values()})


def get_entries_for_crop(crop: str) -> list[tuple[str, DiseaseInfo]]:
    return sorted(
        [(k, v) for k, v in DISEASE_LIBRARY.items() if v.crop == crop],
        key=lambda kv: kv[1].condition,
    )
