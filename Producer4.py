from math import ceil
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pandas as pd
import h5py

K_MER_SIZE = 6
BATCH_SIZE = 4
LAYERS_TO_SAVE = 30

MODEL_PATH = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
OUT_PATH = 'V2_500_multi_embeddings_expanded_CLS_separate_N_labeled_seqs.h5'
expanded_file = 'New_sequences_labeled_with_Ns.csv'

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True)

df = pd.read_csv(expanded_file)
all_sequences = df['sequence'].astype(str).tolist()
labels = df['label'].astype(int).tolist()
print(f'Loaded {len(all_sequences)} expanded sequences')

MAX_SEQUENCES = len(all_sequences)

print('Class distribution:',
      {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})

max_Ns = max(seq.count("N") for seq in all_sequences)
print("Maximum number of Ns in any input sequence is:", max_Ns)

max_length = ceil(max([len(_) for _ in all_sequences]) / K_MER_SIZE + K_MER_SIZE - 1) + max_Ns * 5
# Finds the maximum length sequence in the set, divides by the number of hexamers, and adds five positions for any dangling nucleotides
print(f'Maximum positions is: {max_length}')

layers = {}
cls_dict = {}

for i, j in enumerate(range(0, MAX_SEQUENCES, BATCH_SIZE)):

    print(f'Working on batch {i} at position {j} of all_sequences')
    sequences = all_sequences[j: j + BATCH_SIZE]
    print(f'There are {len(sequences)} sequences in this batch')
    for seq in sequences:
        print(seq)
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
    print(f'Shape of reduced attention mask is {attention_mask[:,1:,:].shape}')

    for k in range(number_of_hidden_states - LAYERS_TO_SAVE, number_of_hidden_states):
        print(f'\tHidden state {k} type is:', type(torch_outs['hidden_states'][k]), f'with shape', torch_outs['hidden_states'][k].shape)

        embeddings = torch_outs['hidden_states'][k].detach().numpy()
        # Grab the data for a particular layer and dump into a numpy array.

        print(f'\tExtracted embedding for layer {k} has shape {embeddings.shape}')

        # Compute mean embeddings per sequence for this layer
        mean_sequence_embeddings = torch.sum(attention_mask[:, 1:, :] * embeddings[:, 1:, :], dim=1) / torch.sum(attention_mask[:, 1:, :], dim=1)
        isolated_CLS = torch.from_numpy(embeddings[:, 0, :].astype(np.float32))
        print(f'\tThe shape of the isolated CLS data is {isolated_CLS.shape}')

        try:
            temp_layer = layers[k]
            layers[k] = torch.cat((temp_layer, mean_sequence_embeddings), dim=0)
            # build a dict with layer keys, adding the new data from the current batch
        except KeyError:
            layers[k] = mean_sequence_embeddings

        try:
            temp_cls = cls_dict[k]
            cls_dict[k] = torch.cat((temp_cls, isolated_CLS), dim=0)
            # build a dict with layer keys, adding the new data from the current batch
        except KeyError:
            cls_dict[k] = isolated_CLS

        print(f'\tShape of mean sequence embeddings for layer {k} is {mean_sequence_embeddings.shape}')

print(f'There are a total of {len(layers)} layers with keys {layers.keys()}')

labels = np.asarray(labels, dtype=np.int32)
with h5py.File(OUT_PATH, "w") as ofh:
    ofh.create_dataset(name="Labels", data=labels)

    for k, v in layers.items():
        layer_name = 'Layer' + str(k)
        layer_data = np.asarray(layers[k], dtype=np.float32)
        ofh.create_dataset(name=layer_name, data=layer_data)
        #print(f'Layer {k} has length of {len(layers[15])}')

    for k, v in layers.items():
        cls_name = 'CLS' + str(k)
        cls_data = np.asarray(cls_dict[k], dtype=np.float32)
        ofh.create_dataset(name=cls_name, data=cls_data)
        # print(f'CLS data for layer {k} has length of {len(layers[15])}')