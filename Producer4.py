from math import ceil
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pandas as pd
import h5py

K_MER_SIZE = 6
BATCH_SIZE = 16
LAYERS_TO_SAVE = 10
MAX_SEQUENCES = 1072
OUT_PATH = "V2_250_multi_embeddings.h5"

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='InstaDeepAI/nucleotide-transformer-v2-250m-multi-species', trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path='InstaDeepAI/nucleotide-transformer-v2-250m-multi-species', trust_remote_code=True)

expanded_file = 'E:/Users/Wombat/Documents/Python/integrase2.csv'
df = pd.read_csv(expanded_file)
all_sequences = df['sequence'].astype(str).tolist()
labels = df['label'].astype(int).tolist()
print(f'Loaded {len(all_sequences)} expanded sequences')
print('Class distribution:',
      {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})

max_length = ceil(max([len(_) for _ in all_sequences]) / K_MER_SIZE + K_MER_SIZE - 1)
# Finds the maximum length sequence in the set, divides by the number of hexamers, and adds five positions for any dangling nucleotides
print(f'Maximum positions is: {max_length}')

# Compute the embedding'

layers = {}

for i, j in enumerate(range(0, MAX_SEQUENCES, BATCH_SIZE)):

    print(f'Working on batch {i} at position {j} of all_sequences')
    sequences = all_sequences[j: j + BATCH_SIZE]
    print(f'There are {len(sequences)} sequences in this batch')
    token_ids = tokenizer.batch_encode_plus(
        sequences,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length'
        )['input_ids']
    # print(token_ids)
    attention_mask = token_ids != tokenizer.pad_token_id
    print(f'Attention mask has type {type(attention_mask)} and shape {attention_mask.shape}')

    torch_outs = model(
        token_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
        )
    # print('Type of raw torch outputs is:', type(torch_outs))
    number_of_hidden_states = len(torch_outs['hidden_states'])

    print(f'Number of hidden states for batch {i} is', number_of_hidden_states)
    attention_mask = attention_mask.unsqueeze(2)
    print(f'Shape of the unsqueezed attention mask is {attention_mask.shape}')

    for k in range(number_of_hidden_states - LAYERS_TO_SAVE, number_of_hidden_states):
        print(f'\tHidden state {k} type is:', type(torch_outs['hidden_states'][k]), f'with shape', torch_outs['hidden_states'][k].shape)

        embeddings = torch_outs['hidden_states'][k].detach().numpy()
        # Grab the data for a particular layer and dump into a numpy array.

        print(f'\tExtracted embedding for layer {k} has shape {embeddings.shape}')

        # Compute mean embeddings per sequence for this layer
        mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)

        try:
            temp_layer = layers[k]
            layers[k] = torch.cat((temp_layer, mean_sequence_embeddings), dim=0)
            # build a dict with layer keys, adding the new data from the current batch
        except KeyError:
            layers[k] = mean_sequence_embeddings

        print(f'\tShape of mean sequence embeddings for layer {k} is {mean_sequence_embeddings.shape}')

print(f'There are a total of {len(layers)} with keys {layers.keys()}')

labels = np.asarray(labels, dtype=np.int32)
with h5py.File(OUT_PATH, "w") as ofh:
    ofh.create_dataset(name="Labels", data=labels)

    for k, v in layers.items():
        layer_name = 'Layer' + str(k)
        layer_data = np.asarray(layers[k], dtype=np.float32)
        ofh.create_dataset(name=layer_name, data=layer_data)
        print(f'Layer {k} has length of {len(layers[15])}')