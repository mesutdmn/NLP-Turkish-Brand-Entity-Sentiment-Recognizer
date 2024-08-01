import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import CustomDataset, TransformerEncoder, load_model_to_cpu

app = FastAPI()



tag2id = {"O": 0, "olumsuz": 1, "nötr": 2, "olumlu": 3, "org": 4}
id2tag = {value: key for key, value in tag2id.items()}

device = torch.device('cpu')
def predict_fonk(model, device, example, tokenizer):
    model.to(device)
    model.eval()
    predictions = []

    encodings_prdict = tokenizer.encode(example)

    predict_texts = [encodings_prdict.tokens]
    predict_input_ids = [encodings_prdict.ids]
    predict_attention_masks = [encodings_prdict.attention_mask]
    predict_token_type_ids = [encodings_prdict.type_ids]
    prediction_labels = [encodings_prdict.type_ids]

    predict_data = CustomDataset(predict_texts, predict_input_ids, predict_attention_masks, predict_token_type_ids,
                                 prediction_labels)

    predict_loader = DataLoader(predict_data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for dataset in predict_loader:
            batch_input_ids = dataset['input_ids'].to(device)
            batch_att_mask = dataset['attention_mask'].to(device)

            outputs = model(batch_input_ids, batch_att_mask)
            logits = outputs.view(-1, outputs.size(-1))  # Flatten the outputs
            _, predicted = torch.max(logits, 1)

            # Ignore padding tokens for predictions
            predictions.append(predicted)

    results_list = []
    entity_list = []
    results_dict = {}
    for i, (token, label, attention) in enumerate(zip(predict_loader.dataset[0]["text"], predictions[0].tolist(),
                                       predict_attention_masks[0])):
        if attention != 0 and label != 0 and label !=4 and token not in [sep for sepx in entity_list for sep in sepx.split()]:
            for next_ones in predictions[0].tolist()[i+1:]:
                i+=1
                if next_ones == 4:
                    token = token +" "+ predict_loader.dataset[0]["text"][i]
                else:break
            if token not in entity_list:
                entity_list.append(token)
                results_list.append({"entity":token,"sentiment":id2tag.get(label)})


    results_dict["entity_list"] = entity_list
    results_dict["results"] = results_list


    return results_dict

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    model = TransformerEncoder()
    model, start_epoch = load_model_to_cpu(model, "model.pth")
    tokenizer = Tokenizer.from_file("tokenizer.json")

    predict_list = predict_fonk(model=model, device=device, example=item.text, tokenizer=tokenizer)

    #Buraya model'in çıktısı gelecek
    #Çıktı formatı aşağıdaki örnek gibi olacak
    return predict_list


if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)