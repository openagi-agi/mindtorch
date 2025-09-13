# 🧠 MindTorch

**Modular neural network library** for building, training, and visualizing machine learning models — from Markov Chains to Transformers — one step at a time.

---

## 🚀 Project Goal

**MindTorch** aims to:

> **Demystify AI architectures by building them from the ground up** — with clear code, modular design, and real-time introspection.

Whether you're learning, debugging, researching, or experimenting with new ideas, MindTorch provides both **classic** and **state-of-the-art** models in a clean and extensible format.

---

## ✅ Key Features

* 📦 **Modular core** — Each model is its own class, cleanly separated and easily swappable.
* ⚙️ **Unified architecture interface** — Consistent API for `forward()` across models.
* 🧩 **No magic** — Every model is implemented transparently, no black boxes.
* 🧠 **Visualization-first** *(coming soon)* — Support for activation tracing, attention maps, and projection of "thoughts".
* 🧪 **Trainer-agnostic** — Training logic lives outside models and supports custom schedulers, hooks, and analytics.

---

## 📚 Models Implemented (So Far)

| Category        | Models             |
| --------------- | ------------------ |
| **Simple**      | Markov Chain, HMM  |
| **Feedforward** | MLP                |
| **Recurrent**   | RNN, LSTM, GRU     |
| **Coming Soon** | GAN, BERT, GPT, T5 |

---

## 📁 Structure

```plaintext
mindtorch/
├── core/              # All model architectures
│   ├── simple.py      # Markov, HMM, etc.
│   ├── mlp.py
│   └── recurrent/
│       ├── rnn.py
│       ├── lstm.py
│       └── gru.py
├── train/             # Training utilities
│   └── recurrent_trainer.py
├── vis/               # (WIP) Live model visualization tools
└── README.md
```

---

## 🧠 Philosophy

This library is built by **engineers who want to see the gears turning**, not just run `model.fit()`. Every model is constructed from first principles with extensibility in mind.

> If it can't be **visualized**, **understood**, or **customized**, it doesn’t belong here.

---

## 🔜 Roadmap

* [x] RNN, LSTM, GRU
* [x] MLP
* [x] Markov & HMM
* [ ] GAN (Basic + SoTA)
* [ ] BERT
* [ ] GPT-style decoder-only
* [ ] T5-style encoder-decoder
* [ ] Autoencoder-based thought projection
* [ ] Visual debugging with attention/activation maps

---

## 🧑‍💻 For Developers

* Install requirements:

  ```bash
  pip install torch rich
  ```

* Use a model:

  ```python
  from mindtorch.core.recurrent import RNN
  model = RNN(input_dim=16, hidden_dim=64, output_dim=10)
  ```

* Training utilities are coming soon in `mindtorch.train`

---

## ⚠️ Disclaimer

This is a **research-first, dev-focused** library — not production-ready. Expect fast iteration, opinionated design, and breaking changes.

---

**Your contributions are helping build a new, transparent future.**
