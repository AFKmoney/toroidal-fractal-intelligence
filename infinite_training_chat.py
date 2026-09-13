"""
Infinite Training with Interactive Chat - Toroidal Fractal Intelligence

This script demonstrates:
1. INFINITE TRAINING - continuous learning without boundaries
2. CONTINUOUS THOUGHT - internal reasoning during training
3. REAL-TIME CHAT - test the model at any point
4. STRUCTURE GROWTH - watch atoms/aggregates/abstractions evolve

Usage:
    python infinite_training_chat.py
"""

import torch
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

# Import our modules
from src.toroidal.model import ToroidalFractalIntelligence
from src.agents.thinker import ThinkerAgent


class InfiniteTrainer:
    """
    Infinite training loop with continuous learning.
    No training/inference boundary - the model learns forever.
    """

    def __init__(self, model, tokenizer, dataset_text, batch_size=16):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # Tokenize entire dataset
        self.tokens = tokenizer.encode(dataset_text)
        print(f"Loaded {len(self.tokens)} tokens from dataset")

        # Create sliding window sequences
        self.sequences = []
        seq_len = 128
        for i in range(0, len(self.tokens) - seq_len, seq_len // 2):
            self.sequences.append(torch.tensor(self.tokens[i:i + seq_len]))

        print(f"Created {len(self.sequences)} training sequences")

        # Training state
        self.step = 0
        self.epoch = 0
        self.running = True

        # Metrics tracking
        self.loss_history = []
        self.atom_history = []
        self.aggregate_history = []
        self.abstraction_history = []

        # Thinker agent for continuous thought
        self.thinker = ThinkerAgent(model, n_thoughts=8)

    def get_batch(self):
        """Get a random batch (infinite loop)."""
        idx = torch.randint(0, len(self.sequences), (self.batch_size,))
        batch = torch.stack([self.sequences[i] for i in idx])
        return batch

    def train_step(self):
        """Single training step with continuous learning."""
        batch = self.get_batch()
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)

        # Forward pass through full pipeline
        output = self.model(token_ids)
        logits = output["logits"]

        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(logits, target_ids)

        # Backward pass (continuous learning)
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=3e-4,
                weight_decay=0.01
            )
        self.optimizer.step()

        # Track metrics
        self.loss_history.append(loss.item())
        self.atom_history.append(len(self.model.atoms))
        self.aggregate_history.append(len(output.get("aggregates", [])))
        self.abstraction_history.append(len(output.get("abstractions", [])))

        self.step += 1
        return loss.item()

    def think(self):
        """Generate internal thoughts based on current state."""
        with torch.no_grad():
            # Get current state representation
            alpha = self.model.state.get_field()
            context = alpha.mean(dim=0).unsqueeze(0)

            # Generate thoughts
            thoughts = self.thinker.think(context, n_steps=3)

            # Integrate thoughts back into state
            enhanced_alpha = self.thinker.integrate_thoughts(thoughts, alpha)

            # Update state
            self.model.state.alpha.data = enhanced_alpha

        return thoughts.detach().cpu().numpy()

    def chat(self, prompt, max_length=100):
        """Generate response to user input."""
        # First, let the model "think" about the prompt
        prompt_ids = self.tokenizer.encode(prompt)
        thoughts = self.thinker.think(
            self.model.state.get_field().mean(dim=0).unsqueeze(0),
            n_steps=5
        )

        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
            )

        return self.tokenizer.decode(generated)

    def save_checkpoint(self, name="checkpoint"):
        """Save model state."""
        path = f"./checkpoints/{name}_{self.step}.pt"
        self.model.save(path)
        return path

    def run_training_loop(self, max_steps=10000, checkpoint_interval=500, chat_interval=100):
        """
        INFINITE TRAINING LOOP.

        This is NOT a fixed training run. The model learns continuously:
        - Every token modifies the computational matter
        - Structures evolve through interaction
        - Abstractions form and consolidate
        - No boundary between training and inference
        """
        print("=" * 70)
        print("TOROIDAL FRACTAL INTELLIGENCE — INFINITE TRAINING MODE")
        print("=" * 70)
        print(f"Starting continuous learning at {datetime.now().isoformat()}")
        print(f"Max steps: {max_steps}")
        print(f"Checkpoint every: {checkpoint_interval} steps")
        print(f"Chat interval: {chat_interval} steps")
        print("=" * 70)
        print()

        start_time = time.time()
        last_checkpoint = 0
        last_chat = 0

        for step in range(1, max_steps + 1):
            # Training step (continuous learning)
            loss = self.train_step()

            # Periodic thinking (continuous thought)
            if step % 10 == 0:
                self.think()

            # Progress logging
            if step % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(self.loss_history[-100:]) / 100
                n_atoms = len(self.model.atoms)
                n_aggs = len(self.model.aggregation.hierarchy[-1]) if self.model.aggregation.hierarchy else 0
                n_abs = len(self.model.abstraction.get_active_abstractions())

                print(f"Step {step:5d}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Atoms: {n_atoms:4d} | "
                      f"Aggs: {n_aggs:3d} | "
                      f"Abs: {n_abs:2d} | "
                      f"Time: {elapsed:.1f}s | "
                      f"Speed: {100/elapsed*100:.1f} steps/sec")

            # Save checkpoint
            if step - last_checkpoint >= checkpoint_interval:
                path = self.save_checkpoint("infinite")
                print(f"  ✓ Saved checkpoint: {path}")
                last_checkpoint = step

            # Enable chat
            if step - last_chat >= chat_interval:
                print("\n" + "=" * 70)
                print("CHATTING MODE - Ask the model anything!")
                print("=" * 70)
                self.interactive_chat()
                print("=" * 70)
                print("Resuming training...")
                print("=" * 70 + "\n")
                last_chat = step

        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total steps: {self.step}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Final loss: {self.loss_history[-1]:.4f}")
        print(f"Final atoms: {len(self.model.atoms)}")
        print(f"Final aggregates: {len(self.model.aggregation.hierarchy) if self.model.aggregation.hierarchy else 0}")
        print(f"Final abstractions: {len(self.model.abstraction.get_active_abstractions())}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Efficiency: {len(self.model.atoms) / max(sum(p.numel() for p in self.model.parameters()), 1):.4f} atoms/parameter")
        print("=" * 70)

    def interactive_chat(self):
        """Interactive chat session."""
        print("\nType 'quit' to exit chat, 'status' for model info, 'think' for internal thoughts")
        print()

        while self.running:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'status':
                    self.print_status()
                    continue
                if user_input.lower() == 'think':
                    thoughts = self.think()
                    print(f"\nInternal thoughts ({len(thoughts)} thoughts generated)")
                    continue

                if user_input:
                    print("\nThinking...", end=" ")
                    response = self.chat(user_input)
                    print(f"\nModel: {response}")

            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def print_status(self):
        """Print current model status."""
        summary = self.model.get_shared_params_summary()
        print(f"\nModel Status:")
        print(f"  Step: {self.step}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Atoms: {summary['n_atoms']}")
        print(f"  Modes: {summary['n_modes']}")
        print(f"  d_model: {summary['d_model']}")
        print(f"  Coupling scale: {summary['coupling_scale']:.4f}")
        print(f"  Energy decay: {summary['energy_decay']:.4f}")
        print(f"  Phase sync: {summary['phase_sync']:.4f}")
        print(f"  Avg loss (last 100): {sum(self.loss_history[-100:])/100:.4f}")
        print()


def main():
    """Main entry point."""
    print("=" * 70)
    print("TOROIDAL FRACTAL INTELLIGENCE — INFINITE TRAINING + CHAT")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model (small for demo, scale up as needed)
    print("Creating model...")
    model = ToroidalFractalIntelligence(
        vocab_size=50257,
        d_model=128,
        n_modes=128,
        n_atoms_max=512,
    )

    # Sample dataset (in production, use Wikipedia/Common Crawl)
    # Using a larger Shakespeare excerpt for better learning
    dataset_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles,
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    For who would bear the whips and scorns of time,
    Th'oppressor's wrong, the proud man's contumely,
    The pangs of dispriz'd love, the law's delay,
    The insolence of office, and the spurns
    That patient merit of th'unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? Who would fardels bear,
    To grunt and groan under a weary life,
    But that the dread of something after death,
    The undiscovere'd country, from whose bourn
    No traveller returns, puzzles the will,
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    Thus conscience does make cowards of us all,
    And thus the native hue of resolution
    Is sicklied o'er with the pale cast of thought,
    And enterprises of great pith and moment
    With this regard their currents turn awry
    And lose the name of action.

    What light through yonder window breaks?
    It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon,
    Who is already sick and pale with grief,
    That thou her maid art far more fair than she.
    Be not her maid, since she is envious;
    Her vestal livery is but sick and green
    And none but fools do wear it; cast it off.
    It is my lady, O, it is my love!
    O, that she knew she were! She speaks yet she says nothing.
    O, speak again, bright angel, for thou art
    As glorious to this night, being o'er my head,
    As is a winged messenger of heaven
    Unto the white-upturn'd wondering eyes
    Of mortals that fall back to gaze on him
    When he bestrides the lazy-pacing clouds
    And sails upon the bosom of the air.

    O Romeo, Romeo! wherefore art thou Romeo?
    Deny thy father and refuse thy name;
    Or, if thou wilt not, be but sworn my love
    And I'll no longer be a Capulet.
    'Tis but thy name that is my enemy;
    Thou art thyself, though not a Montague.
    What's Montague? it is nor hand, nor foot,
    Nor arm, nor face, nor any other part
    Belonging to a man. O, be some other name!
    What's in a name? that which we call a rose
    By any other name would smell as sweet;
    So Romeo would, were he not Romeo call'd,
    Retain that dear perfection which he owes
    Without that title. Romeo, doff thy name,
    And for that name, which is no part of thee,
    Take all myself.

    Good night, good night! Parting is such sweet sorrow,
    That I shall say good night till it be morrow.
    Sleep dwell upon thine eyes, peace in thy breast!
    Would I were sleep and peace, so sweet to rest!
    Hence will I to my ghostly father's cell,
    His help to crave, and my dear hap to tell.

    For never was a story of more woe,
    Than this of Juliet and her Romeo.

    All the world's a stage,
    And all the men and women merely players;
    They have their exits and their entrances,
    And one man in his time plays many parts,
    His acts being seven ages.

    Tomorrow, and tomorrow, and tomorrow,
    Creeps in this petty pace from day to day,
    To the last syllable of recorded time,
    And all our yesterdays have lighted fools
    The way to dusty death. Out, out, brief candle!
    Life's but a walking shadow, a poor player,
    That struts and frets his hour upon the stage,
    And then is heard no more. It is a tale
    Told by an idiot, full of sound and fury,
    Signifying nothing.

    We are such stuff as dreams are made on,
    And our little life is rounded with a sleep.

    The course of true love never did run smooth.

    Lord, what fools these mortals be!

    Though this be madness, yet there is method in 't.

    There are more things in heaven and earth, Horatio,
    Than are dreamt of in your philosophy.

    The lady doth protest too much, methinks.

    All that glisters is not gold.

    Brevity is the soul of wit.

    Neither a borrower nor a lender be.

    This above all: to thine own self be true.
    """

    # Create trainer
    trainer = InfiniteTrainer(model, tokenizer, dataset_text, batch_size=16)

    # Run infinite training with chat
    try:
        trainer.run_training_loop(
            max_steps=5000,
            checkpoint_interval=500,
            chat_interval=200,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        trainer.save_checkpoint("interrupted")
        print("Saved checkpoint before exit.")

    print("\nThank you for using Toroidal Fractal Intelligence!")
    print("Remember: training never truly ends. You can always continue.")


if __name__ == "__main__":
    main()
