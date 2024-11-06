#### 2024 Teknofest, Turkish Natural Language Processing Competition Natural Language Processing Scenario.
#### This project was prepared for the Teknofest **Acıkhack2024TDDİ** competition.
---
## Model Integration with FastAPI

This project aims to deploy a model trained with PyTorch in a web-based application using FastAPI.
Additionally, you can test the deployed model using this Streamlit link: [Streamlit](https://huggingface.co/spaces/We-Bears/Turkish-NER-Sentiment-Streamlit)

## Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Downloading the Model and Tokenizer
Since the model size exceeds GitHub's 100MB limit, it has been uploaded to Google Drive.
Download the trained model and tokenizer from the Google Drive link below and place them in the same directory as `app.py`:

[Download the Model and Tokenizer](https://drive.google.com/drive/folders/1u6J98lXvI-iXySYQgAZ053B8V3jPPILN?usp=sharing)

## Running the Application

Start your FastAPI application using the command below:

```bash
python app.py
```

## Swagger Interface

To access the Swagger interface provided by FastAPI, visit the following address in your browser:

```
http://127.0.0.1:8000/docs
```

Through this interface, you can test your API and access the documentation.

## Project Structure

```plaintext
.
├── app.py              
├── model.py            
│── model.pth
│── tokenizer.json
|── requirements.txt
└── README.md         
```

## Retraining

Data and Tokenizer are shared in the `train/data` directory. To retrain the model, follow the steps in the `train.ipynb` notebook located in the `train` directory. Once training is complete, the model will output the results.

## License

This project is licensed under the Apache 2.0 License. For more information, see the LICENSE file.

## Data Sources
https://huggingface.co/datasets/kmkarakaya/turkishReviews-ds

https://tr.wikipedia.org/

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
### 📺 Demo Video

https://github.com/user-attachments/assets/89d7ff5e-4b96-4220-8bc0-fc4f6d6a1d4f



