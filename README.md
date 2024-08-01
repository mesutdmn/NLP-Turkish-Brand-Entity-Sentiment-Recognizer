#### 2024 Teknofest, Türkçe Doğal Dil İşleme Yarışması Doğal Dil İşleme Senaryosu.
#### Bu proje Teknofest **Acıkhack2024TDDİ** yarışması için hazırlanmıştır.
---
## FastAPI ile Model Entegrasyonu

Bu proje, PyTorch ile eğitilmiş bir modeli FastAPI kullanarak web tabanlı bir uygulamada sunmayı amaçlamaktadır.
Aynı zamanda, bu streamlit linkinden deploy edilmiş modeli test edebilirsiniz: [Streamlit](https://teknofestnlpmodel.streamlit.app/)

## Kurulum

Öncelikle, gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## Model ve Tokenizer'ı İndirme
800mb model boyutu github limitleri olan 100mb değerini aştığı için googledrive yüklenmiştir.
Eğitilmiş model ve tokenizer'ı aşağıdaki Google Drive linkinden indiriniz ve `app.py` ile aynı dizine yerleştiriniz:

[Model ve Tokenizer'ı İndir](https://drive.google.com/drive/folders/1u6J98lXvI-iXySYQgAZ053B8V3jPPILN?usp=sharing)

## Uygulamayı Çalıştırma

Aşağıdaki komutu kullanarak FastAPI uygulamanızı başlatın:

```bash
python app.py
```

## Swagger Arayüzü

FastAPI tarafından sunulan Swagger arayüzüne erişmek için tarayıcınızda şu adresi ziyaret edin:

```
http://127.0.0.1:8000/docs
```

Bu arayüz üzerinden API'nizi test edebilir ve dokümantasyona erişebilirsiniz.

## Proje Yapısı

```plaintext
.
├── app.py              
├── model.py            
│── model.pth
│── tokenizer.json
|── requirements.txt
└── README.md         
```

## Yeniden Eğitim

Veri ve Tokenizer train/data dizininde paylaşılmıştır, yeniden eğitim yapmak için train dizinindeki train.ipynb notebook dosyasındaki adımlar izlenebilir, eğitim tamamlandığında model çıktısı verecektir.

## Lisans

Bu proje Apache 2.0 Lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakınız.


## 👨‍👩‍👧‍👦We Bears Team Members 

- ### 🙋‍♂️Burhan Yıldız

<a target="_blank" href="https://www.linkedin.com/in/burhanyildiz/"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white"></img></a>
<a target="_blank" href="https://www.kaggle.com/yldzburhan"><img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white"></img></a>
<a target="_blank" href="https://medium.com/@yildizburhan"><img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></img></a>

- ### 🙋‍♂️Hüseyin Baytar

<a target="_blank" href="https://www.linkedin.com/in/huseyinbaytar/"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white"></img></a>
<a target="_blank" href="https://www.kaggle.com/huseyinbaytar"><img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white"></img></a>
<a target="_blank" href="https://medium.com/@huseyinbaytar"><img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></img></a>

- ### 🙋‍♂️Mesut Duman

<a target="_blank" href="https://www.linkedin.com/in/mesut-duman/"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white"></img></a>
<a target="_blank" href="https://www.kaggle.com/dumanmesut"><img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white"></img></a>
<a target="_blank" href="https://medium.com/@dumanmesut"><img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></img></a>

---
