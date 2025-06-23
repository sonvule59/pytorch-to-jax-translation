2.24.0

output_code:
python
import jax
import jax.numpy as jnp
from model import cosine_sim
from evaluation import encode_data

def select_random_jax(r, model, train_loader):
    if r == 0:
        return jnp.array(random.sample(range(0, len(train_loader.dataset)), 1280), dtype=jnp.int64)
    else:
        return jnp.array(random.sample(range(0, len(train_loader.dataset)), 128), dtype=jnp.int64)

def select_margin_jax(r, model, train_loader, primary="image"):
    if r == 0:
        return jnp.array(random.sample(range(0, len(train_loader.dataset)), 1280), dtype=jnp.int64)
    else:
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (cap_embs, img_embs)

        scores = []
        for i in range(0, len(primary_embs), 128):
            batch_range = min(128, len(primary_embs) - i)
            primary_batch = jnp.array(primary_embs[i:i + batch_range], dtype=jnp.float32)
            primary_batch = jnp.as_tensor(primary_batch, jax.float32)

            primary_secondary_distances = jnp.array([], dtype=jnp.float32)
            for j in range(0, len(secondary_embs), 128):
                batch_range2 = min(128, len(secondary_embs) - j)
                secondary_batch = jnp.array(secondary_embs[j:j + batch_range2], dtype=jnp.float32)
                secondary_batch = jnp.as_tensor(secondary_batch, jax.float32)

                cosine_dist = primary_batch.mm(secondary_batch.t())
                if j == 0:
                    primary_secondary_distances = jnp.cat(cosine_dist, axis=1)
                else:
                    primary_secondary_distances = jnp.cat((primary_secondary_distances, cosine_dist), axis=1)

            distances_top2 = jnp.abs(jnp.topk(primary_secondary_distances, 2, axis=1, largest=False)[1])
            margin = jnp.abs(distances_top2[:, 0] - distances_top2[:, 1])
            scores.extend(margin.cpu().numpy())  # This line may not work in pure JAX; scores should be jnp array

        best_n_indices = jnp.array([n[0] for n in jnp.nsmallest(128, enumerate(scores), key=lambda x: x[1])])
        return best_n_indices

def select_caption_jax(r, model, train_loader, primary="image"):
    if r == 0:
        return jnp.array(random.sample(range(0, len(train_loader.dataset)), 1280), dtype=jnp.int64)
    else:
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (cap_embs, img_embs)

        scores = []
        for i in range(0, len(primary_embs), 5):
            batch_range = min(5, len(primary_embs) - i)
            primary_batch = jnp.array(primary_embs[i:i + batch_range], dtype=jnp.float32)
            primary_batch = jnp.as_tensor(primary_batch, jax.float32)

            avg_distance = get_avg_distance(primary_batch)
            scores.append(avg_distance)

        best_n_indices = jnp.array([n[0] for n in jnp.nsmallest(128, enumerate(scores), key=lambda x: x[1])])
        print(best_n_indices)
        #worst_n_indices = jnp.array([n[0] for n in jnp.nlargest(128, enumerate(scores), key=lambda x: x[1])])

        return best_n_operations_indices

def get_avg_distance(primary_batch):
    total_cap = jnp.sum(primary_batch * jnp.ones_like(primary_batch), axis=-1)
    avg_cap = jnp.div(total_cap, len(primary_batch))
    cosine_dist = avg_cap.unsqueeze(1).mm(primary_batch.t())
    average_distance = jnp.sum(cosine_dist) / primary_batch.size(-1)
    return average_distance

def select_uncertainty_jax(r, model, train_loader, primary="image"):
    if r == 0:
        return jnp.array(random.sample(range(0, len(train_loader.dataset)), 1280), dtype=jnp.int64)
    else:
        model.val_start()
        img_embs, cap_embs = encode_data(model, train_loader)
        primary_embs, secondary_embs = (img_embs, cap_embs) if primary == "image" else (