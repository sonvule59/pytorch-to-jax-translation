import jax.numpy as jnp
from jax import concat, range

def pad_ids(jids, jmask, jtid, max_length, jpad_token, jmask_pd_with_zero, jpad_tid_seg_id, jpad_on_left=False):
    pad_len = max_length - len(jids)
    if jpad_on_left:
        jids = jnp.pad(jids, (0, pad_len, 0), mode='pre')
        jmask = jnp.pad(jmask, (0, pad_len, 0), mode='post')
        jtid = jnp.pad(jtid, (0, pad_len, 0), mode='post')
    else:
        jids = jnp.pad(jids, (0, pad_len, 0), mode='post')
        jmask = jnp.pad(jmask, (0, pad_len, 0), mode='post')
        jtid = jnp.pad(jtid, (0, pad_len, 0), mode='post')
    return jids, jmask, jtid

def dual_process_fn(line, i, tokenizer, args):
    jfeatures = []
    jcells = line.split("\t")
    if len(jcells) == 2:
        jqid = jnp.int32(i)
        jmask_pd_with_zero = True
        jpad_tid_seg_id = 0

        text = jcells[1].strip()
        jinput_ids_a = tokenizer.encode(
            text, add_special_tokens=True, max_length=args.max_seq_length, drop_tokens=True, truncation_token=tokenizer.eos_token,
            return_tensors='jax')
        jinput_ids_a = jinput_ids_a['input_ids']
        jtoken_type_ids_a = [0] * len(jinput_ids_a)
        jattention_mask_a = [
            1 if jmask_pd_with_zero else 0 for _ in jinput_ids_a
        ]
        jinput_ids_a, jattention_mask_a, jtoken_type_ids_a = pad_ids(
            jinput_ids_a, jattention_mask_a, jtoken_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, jmask_pd_with_zero, jpad_tid_seg_id, jpad_on_left)
        jfeatures.extend([
            jnp.tensor(jinput_ids_a, dtype=jnp.int64),
            jnp.tensor(jattention_mask_a, dtype=jnp.bool_)
        ])
        jfeatures.append(jqid)
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 2.".format(str(len(jcells))))
    return jfeatures

def triple_process_fn(line, i, tokenizer, args):
    jfeatures = []
    jcells = line.split("\t")
    if len(jcells) == 3:
        jmask_pd_with_zero = True
        jpad_tid_seg_id = 0

        for jtext in jcells:
            jinput_ids_a = tokenizer.encode(
                jtext.strip(), add_special_tokens=True, max_length=args.max_seq_length, drop_tokens=True, truncation_token=tokenizer.eos_token,
                return_tensors='jax')
            jinput_ids_a = jinput_ids_a['input_ids']
            jtoken_type_ids_a = [0] * len(jinput_ids_a)
            jattention_mask_a = [
                1 if jmask_pd_with_zero else 0 for _ in jinput_ids_a
            ]
            jinput_ids_a, jattention_mask_a, jtoken_type_a = pad_ids(
                jinput_ids_a, jattention_mask_a, jtoken_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, jmask_pd_with_zero, jpad_tid_seg_id, jpad_on_left)
            jfeatures.extend([
                jnp.tensor(jinput_ids_a, dtype=jnp.int64),
                jnp.tensor(jattention_mask_a, dtype=jnp.bool_)
            ])
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(jcells))))
    return jfeatures

def dual_to_triple_to_ndp(processor_fn, triple_proc_fn, dual_process_fn):
    def ndp_fn(line, i, **kwargs):
        jfeatures = triple_proc_fn(line, i, kwargs)
        dual_features = dual_process_fn(line, i, kwargs)
        return jfeatures + dual_features
    return ndp_fn

# Example usage:
# jlines = ["<pad> 123...",...]
# outputs = jax.vmap(ndp_fn, jlines)

# However, the above approach may not be optimal. 
# Alternative approach: create a unified function that handles all line types.
# But due to time constraints, the current approach is implemented.

# To handle different line types uniformly, we can create a generic function.
def jax_line_processor(line, i, tokenizer, args):
    if b'"query,passage"' in