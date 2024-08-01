import streamlit as st
import subprocess
import base64
st.set_page_config(layout="wide",page_title="TeknoFest We Bears NLP Competition", page_icon="./streamlit/media/3bears.ico")
api_key = st.secrets["Hugging_key"]

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background = get_base64("./streamlit/media/background.jpg")

with open("./streamlit/style/style.css", "r") as style:
    css=f"""<style>{style.read().format(background=background)}</style>"""
    st.markdown(css, unsafe_allow_html=True)

left, middle, right = st.columns([1,1.5,1])
main, comps , result = middle.tabs([" ", " ", " "])
with main:
    example = st.text_area(label='Metin Kutusu: ').strip().replace("\n", " ").replace("\"", " ").replace("'"," ")

    if st.button("Predict"):
        try:
            command = [
                'curl', '-X', 'POST', 'https://mesutdmn-teknofestnermodel.hf.space/predict/',
                '-H', f'Authorization: Bearer {api_key}',
                '-H', 'Content-Type: application/json',
                '-H', 'accept: application/json',
                '--data', '{"text": "' + example + '"}'
            ]

            result = subprocess.run(command, capture_output=True, text=True)
            st.write(eval(result.stdout))

        except Exception as e:
            st.write("Marka Adı Bulunamadı")
            st.write(e)