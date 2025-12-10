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

# Choose device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    try:
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name (0):", torch.cuda.get_device_name(0))
    except Exception:
        pass

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True)

# Move model to device
model.to(device)
model.eval()

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

    # Tokenize (returns CPU tensors) and move tensors to device
    token_ids = tokenizer.batch_encode_plus(
        sequences,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length'
    )['input_ids'].to(device)

    # attention mask (bool -> float) on same device as token_ids
    attention_mask = (token_ids != tokenizer.pad_token_id).to(device)
    print(f'Attention mask has type {type(attention_mask)} and shape {attention_mask.shape}')

    # Convert attention mask to float and add the feature dimension for broadcasting
    attention_mask_float = attention_mask.unsqueeze(2).float()
    print(f'Shape of the unsqueezed attention mask is {attention_mask_float.shape}')
    print(f'Shape of reduced attention mask is {attention_mask_float[:,1:,:].shape}')

    # Run model (inputs must be on the same device as the model)
    with torch.no_grad():
        torch_outs = model(
            token_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

    number_of_hidden_states = len(torch_outs['hidden_states'])
    print(f'Number of hidden states for batch {i} is', number_of_hidden_states)

    for k in range(number_of_hidden_states - LAYERS_TO_SAVE, number_of_hidden_states):
        print(f'\tHidden state {k} type is:', type(torch_outs['hidden_states'][k]), f'with shape', torch_outs['hidden_states'][k].shape)

        # Keep tensors on device while computing; detach to avoid grads
        embeddings = torch_outs['hidden_states'][k].detach()  # shape: (batch, seq_len, hidden_dim)
        print(f'\tExtracted embedding for layer {k} has shape {embeddings.shape}')

        # Compute mean embeddings per sequence for this layer using attention mask
        numerator = torch.sum(attention_mask_float[:, 1:, :] * embeddings[:, 1:, :], dim=1)
        denominator = torch.sum(attention_mask_float[:, 1:, :], dim=1)
        # avoid division by zero
        denominator = torch.clamp(denominator, min=1e-9)
        mean_sequence_embeddings = numerator / denominator

        # CLS token embeddings (isolate index 0)
        isolated_CLS = embeddings[:, 0, :].detach().float()

        print(f'\tThe shape of the isolated CLS data is {isolated_CLS.shape}')

        # Accumulate per-layer tensors (keep on device)
        if k in layers:
            layers[k] = torch.cat((layers[k], mean_sequence_embeddings), dim=0)
        else:
            layers[k] = mean_sequence_embeddings

        if k in cls_dict:
            cls_dict[k] = torch.cat((cls_dict[k], isolated_CLS), dim=0)
        else:
            cls_dict[k] = isolated_CLS

        print(f'\tShape of mean sequence embeddings for layer {k} is {mean_sequence_embeddings.shape}')

print(f'There are a total of {len(layers)} layers with keys {layers.keys()}')

labels = np.asarray(labels, dtype=np.int32)

# When saving, move tensors to CPU and convert to numpy
with h5py.File(OUT_PATH, "w") as ofh:
    ofh.create_dataset(name="Labels", data=labels)

    for k, v in layers.items():
        layer_name = 'Layer' + str(k)
        # v is a torch.Tensor (possibly on GPU); move to CPU and convert
        layer_data = v.detach().cpu().numpy().astype(np.float32)
        ofh.create_dataset(name=layer_name, data=layer_data)

    for k, v in cls_dict.items():
        cls_name = 'CLS' + str(k)
        cls_data = v.detach().cpu().numpy().astype(np.float32)
        ofh.create_dataset(name=cls_name, data=cls_data)