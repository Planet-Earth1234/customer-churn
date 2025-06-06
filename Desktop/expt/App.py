import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import timm

# Class names for the model
li = ['aam_panna', 'adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_pattice', 'aloo_shimla_mirch', 'aloo_tikki', 'aloo_vadi', 'anarsa', 'appe', 'ariselu', 'baingan_bharta', 'bandar_laddu', 'barfi', 'basundi', 'bebinca', 'beetroot_tikki', 'besan_ladoo', 'bhakarwadi', 'bhapa_doi', 'bhatura', 'bhel_puri', 'bhindi_masala', 'biryani', 'bisi_bele_bath', 'bombay_aloo', 'boondi', 'bread_pakora', 'butter_chicken', 'capsicum_curry', 'chaas', 'chai', 'chakli', 'chak_hao_kheer', 'cham_cham', 'chana_chaat', 'chana_masala', 'chapati', 'chawal', 'cheela', 'cheese_naan', 'chicken_65', 'chicken_chilli', 'chicken_korma', 'chicken_lolipop', 'chicken_razala', 'chicken_seekh_kebab', 'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'chilli_cheese_toastie', 'chole_bhature', 'chowmein', 'coconut_chutney', 'corn_cheese_balls', 'daal_bhaati_churma', 'daal_puri', 'dabeli', 'dahi_bhalla', 'dahi_chaat', 'dal_makhani', 'dal_rice', 'dal_tadka', 'dharwad_pedha', 'dhokla', 'doodhpak', 'double_ka_meetha', 'dum_aloo', 'egg_bhurji', 'egg_curry', 'egg_fried_rice', 'fafda', 'falafel', 'falooda', 'fara', 'fish_curry', 'frankies', 'fruit_custard', 'gajar_ka_halwa', 'galouti_kebab', 'gavvalu', 'ghevar', 'gobi_manchurian', 'gujiya', 'gulab_jamun', 'gulgula', 'halwa', 'handvo', 'hara_bhara_kabab', 'idli_sambhar', 'imarti', 'jalebi', 'jeera_rice', 'kachori', 'kadai_chicken', 'kadai_paneer', 'kadhi_chawal', 'kadhi_pakoda', 'kajjikaya', 'kaju_katli', 'kakinada_khaja', 'kalakand', 'karela_bharta', 'kathal_curry', 'kathi_roll', 'keema', 'khandvi', 'kheer', 'khichdi', 'kofta', 'kosha_mangsho', 'kulcha', 'kulfi', 'lassi', 'ledikeni', 'lemon_rasam', 'lemon_rice', 'litti_chokha', 'lyangcha', 'maach_jhol', 'makki_di_roti_sarson_da_saag', 'malai_chaap', 'malai_kofta', 'malapua', 'mango_chutney', 'mango_lassi', 'matar_paneer', 'misal_pav', 'misi_roti', 'misti_doi', 'modak', 'momos', 'moong_dal_halwa', 'motichoor_ladoo', 'mushroom_curry', 'mysore_masala_dosa', 'mysore_pak', 'naan', 'nankhatai', 'navratan_korma', 'neer_dosa', 'pakora', 'palak_paneer', 'paneer_butter_masala', 'paneer_chilli', 'paneer_lababdar', 'paneer_tikka', 'pani_puri', 'paratha', 'pav_bhaji', 'peda', 'phirni', 'pithe', 'poha', 'pongal', 'poornalu', 'pootharekulu', 'prawn_curry', 'pulao', 'pumpkin_sabzi', 'puran_poli', 'qubani_ka_meetha', 'rabri', 'ragda_patties', 'raita', 'rajma_chawal', 'rasam', 'rasgulla', 'ras_malai', 'rogan_josh', 'roti', 'sabudana_vada', 'sambhar_vada', 'samosa', 'samosa_chaat', 'sandesh', 'sevayian', 'sev_puri', 'shahi_paneer', 'shahi_tukda', 'shankarpali', 'sheera', 'sheer_korma', 'shrikhand', 'shwarma', 'sohan_halwa', 'sohan_papdi', 'soya_chap_masala', 'sukat_chutney', 'sutar_feni', 'taak', 'tamarind_rice', 'tandoori_chicken', 'thalipeeth', 'thandai', 'thepla', 'tikki_chaat', 'undhiyu', 'unni_appam', 'upma', 'uttapam', 'vada_pav', 'veg_cutlet', 'veg_kolhapuri', 'vindaloo']

li = [name.replace("_", " ").title() for name in li]

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# Define the custom model class
class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = nn.Linear(self.model.num_features, 206)  # 206 classes

    def forward(self, x):
        return self.model(x)

# Load model weights
@st.cache_resource
def load_model():
    model = CustomEfficientNet().to(device)
    model_weights_path = r"C:\Users\Yash\Desktop\Expt_react\backend\model_weights.pth"  # Update with your model's path
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title("Indian Food Classification")
st.write("Upload an image of Indian food, and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    # Display the predicted class
    predicted_label = li[predicted_class.item()]
    st.write(f"**Predicted Class:** {predicted_label}")
