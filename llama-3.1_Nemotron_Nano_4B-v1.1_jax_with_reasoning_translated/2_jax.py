2.24.0

Convert PyTorch to JAX code equivalently but correctly.
output
python
import jax
import jax.numpy as jnp
import collections

random.seed(42)

class ReplayMemory_jax:

    def __init__(
        self,
        capacity,
        device,
        history=4,
        discount=0.99,
        multi_step=3,
        priority_weight=0.4,
        priority_exponent=0.5
    ):
        super().__init__()
        self.capacity = capacity
        self.device = device
        self.history = history
        self.discount = discount
        self.multi_step = multi_step
        self.priority_weight = priority_weight
        self.priority_exponent = priority_exponent

        self.timestep = 0
        self.transitions = jax.jit_compile(jax.nn.Sequential(
            jax.jit.compile(jax.tfa.mlp(
                jax.jit.vramf(
                    jax.jit.rhs_add,
                    (self.history, self.capacity),
                    (jnp.arange(self.history), -jnp.inf)
                ),
                self.history * self.history, jax.jit.rhs_add
            )),
            jax.jitz.banded_linear(
                jax.jit.rhs_add,
                (self.history, self.history),
                (jnp.arange(self.history), jnp.arange(self.history)),
                bias=0.0, stddev=1.0 / jnp.sqrt(self.history)
            )
        ))).to(jax.device_map.device(self.device)

    def __iter__( JITMode ):
        super().__iter__(JITMode)
        self.current_index = jnp.zeros(1, dtype=jnp.int64, device=self.device)
        return self

    def __next__( self, JITMode ):
        if self.current_index == self.capacity:
            raise StopIteration(JITMode)

        state_stack = jnp.empty((self.history,), dtype=jnp.float32, device=self.device)
        state_stack[-1] = self.transitions.get(self.current_index).state

        previous_timestep = jnp.zeros(1, dtype=jnp.int64, device=self.device)
        previous_timestep[0] = self.transitions.data[self.current_index].timestep

        for t in jnp.arange(self.history - 1, -1, -1):
            if previous_timestep[0] == 0:
                state_stack[t] = self.transitions.data[blank_transition.timestep].state
            else:
                state_stack[t] = self.transitions.data[self.current_index + t - self.history + 1].state
                previous_timestep[0] -= 1

        state = jnp.stack(state_stack, axis=0).to(dtype=jnp.float32, device=self.device).div_(255)

        self.current_index += jnp.ones(1, dtype=jnp.int64, device=self.device)
        return state

    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=jnp.uint8, device=self.device)

        transition = self.transitions.update(
            jnp.ones((1,), dtype=jnp.int64, device=self.device),
            Transition(self.timestep, state, action, reward, terminal == 1)
        )

        self.timestep = 0 if terminal else self.timing + 1

    def sample(self, batch_size):
        priority_total = self.transitions.total_weight()
        jv = jnp.vramf(
            jax.jit.rhs_add,
            (self.capacity, batch_size),
            (jnp.arange(self.capacity), jnp.arange(batch_size)),
            jax.jit.rhs_mul
        )
        jv = jv * (jnp.ones(self.capacity) / jv[0])
        jv = jv.to(dtype=jnp.float32, device=self.device)

        segment = jv[self.current_index:self.current_index + batch_size]
        probs = jax.jit.compile(jax.tfa.lte_permutation(
            jax.jit.rhs_add,
            (batch_size,),
            jnp.ones(batch_size)
        ))(segment / segment.max())

        probs = probs.to(dtype=jnp.float32, device=self.device)
        probs = probs / probs.sum()

        batch = [
            self._get_sample_from_segment(probs, i) for i in jnp.arange(batch_size)
        ]

        probs, indices, tree_indices, states, actions, returns, next_states, nonterminals = zip(*batch)
        probs = jnp.array(probs, dtype=jnp.float32, device=self.device)
        states, next_states = jnp.stack(states, axis=0), jnp.stack(next_states, axis=0)
        actions, returns, nonterminals = jnp.concatenate([actions, returns, nonterminals])

        capacity = self.capacity if self.transitions.full else self.transitions.index