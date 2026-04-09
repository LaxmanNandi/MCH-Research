import json

with open("C:/Users/barla/mch_experiments/data/together_models.json", encoding='utf-8') as f:
    data = json.load(f)

# Find relevant models
print(f"Data type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
if isinstance(data, dict) and 'data' in data:
    models_list = data['data']
elif isinstance(data, list):
    models_list = data
else:
    models_list = []

keywords = ['deepseek', 'llama-4', 'maverick', 'mistral']
for m in sorted(models_list, key=lambda x: x.get('id', '')):
    model_id = m.get('id', '').lower()
    if any(k in model_id for k in keywords):
        print(f"{m['id']}")
