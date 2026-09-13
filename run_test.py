"""
Run test directly without import issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import json
from pathlib import Path
from datetime import datetime

# Import modules directly
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.tokenizer import ToroidalTokenizer

# Tiny Shakespeare dataset (excerpt)
SHAKESPEARE_TEXT = """
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
And lose the name of action. Soft you now,
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.

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
Shall I hear thee, Romeo, assume'dst thy father's name?
Romeo, deny thy father and refuse thy name;
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

But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
And her eyes shine like the stars in the heavens.
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

Good night, good night! Parting is such sweet sorrow,
That I shall say good night till it be morrow.
Sleep dwell upon thine eyes, peace in thy breast!
Would I were sleep and peace, so sweet to rest!
Hence will I to my ghostly father's cell,
His help to crave, and my dear hap to tell.

For never was a story of more woe,
Than this of Juliet and her Romeo.
"""


def create_shakespeare_dataset(text, tokenizer, max_length=128):
    """Create dataset from Shakespeare text."""
    tokens = tokenizer.encode(text)
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length // 2):
        seq = tokens[i:i + max_length]
        if len(seq) == max_length:
            sequences.append(torch.tensor(seq))
    print(f"Created {len(sequences)} training sequences")
    return sequences


def main():
    print("=" * 70)
    print("TOROIDAL FRACTAL INTELLIGENCE — TINY SHAKESPEARE TEST")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Initialize
    tokenizer = ToroidalTokenizer("gpt2")
    model = ToroidalFractalIntelligence(
        vocab_size=50257,
        d_model=128,
        n_modes=128,
        n_atoms_max=256,
    )

    # Create dataset
    print("Creating dataset...")
    sequences = create_shakespeare_dataset(SHAKESPEARE_TEXT, tokenizer, max_length=64)

    if len(sequences) == 0:
        print("ERROR: No sequences created!")
        return

    # Training config
    batch_size = 8
    max_steps = 500
    learning_rate = 1e-3
    log_interval = 50
    save_interval = 250

    # Create dataloader
    def data_iterator():
        while True:
            indices = torch.randperm(len(sequences))
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = torch.stack([sequences[idx] for idx in batch_indices])
                yield batch

    dataloader = data_iterator()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training metrics
    metrics_history = []
    start_time = datetime.now()

    print(f"\nTraining for {max_steps} steps...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Training loop
    for step in range(1, max_steps + 1):
        batch = next(dataloader)
        token_ids = batch[:, :-1]  # Input: all but last
        target_ids = batch[:, 1:]  # Target: all but first

        # Flatten to batch of single tokens
        token_ids = token_ids.reshape(-1)
        target_ids = target_ids.reshape(-1)

        metrics = model.train_step(token_ids, target_ids, optimizer)
        metrics['step'] = step
        metrics['loss_raw'] = metrics['loss']

        metrics_history.append(metrics)

        if step % log_interval == 0:
            avg_loss = sum(m['loss'] for m in metrics_history[-log_interval:]) / log_interval
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Step {step:4d}/{max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Atoms: {metrics['n_atoms']:3d} | "
                  f"Aggs: {metrics['n_aggregates']:2d} | "
                  f"Abs: {metrics['n_abstractions']:2d} | "
                  f"Time: {elapsed:.1f}s")

        if step % save_interval == 0:
            checkpoint_path = f"./checkpoints/shakespeare_step_{step}.pt"
            model.save(checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    # Final metrics
    final_metrics = metrics_history[-1]
    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss: {final_metrics['loss']:.4f}")
    print(f"Final atoms: {final_metrics['n_atoms']}")
    print(f"Final aggregates: {final_metrics['n_aggregates']}")
    print(f"Final abstractions: {final_metrics['n_abstractions']}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Steps/sec: {max_steps / total_time:.2f}")

    # Save final model
    model.save("./checkpoints/shakespeare_final.pt")
    print("\n✓ Saved final model: ./checkpoints/shakespeare_final.pt")

    # Generate text
    print("\n" + "=" * 70)
    print("TEXT GENERATION TESTS")
    print("=" * 70)

    test_prompts = [
        "To be or not to",
        "Romeo, Romeo",
        "What light through",
        "Good night, good",
    ]

    generations = {}
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt)
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                max_length=100,
                temperature=0.8,
                top_k=50,
            )
        text = tokenizer.decode(generated)
        generations[prompt] = text
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Response: \"{text[:150]}...\"")

    # Compute efficiency metrics
    n_params = sum(p.numel() for p in model.parameters())
    n_atoms = len(model.atoms)

    efficiency = {
        "atoms_per_parameter": n_atoms / max(n_params, 1),
        "total_parameters": n_params,
        "final_atoms": n_atoms,
        "avg_atoms_per_step": n_atoms / max_steps,
    }

    # Aggregate metrics
    avg_loss = sum(m['loss'] for m in metrics_history) / len(metrics_history)
    avg_aggregates = sum(m['n_aggregates'] for m in metrics_history) / len(metrics_history)
    avg_abstractions = sum(m['n_abstractions'] for m in metrics_history) / len(metrics_history)

    aggregate_metrics = {
        "avg_loss": avg_loss,
        "final_loss": final_metrics['loss'],
        "avg_atoms": sum(m['n_atoms'] for m in metrics_history) / len(metrics_history),
        "max_atoms": max(m['n_atoms'] for m in metrics_history),
        "avg_aggregates": avg_aggregates,
        "avg_abstractions": avg_abstractions,
        "consolidation_count": final_metrics['consolidation_count'],
    }

    # Save results
    results = {
        "experiment": "tiny_shakespeare",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "d_model": 128,
            "n_modes": 128,
            "n_atoms_max": 256,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "training": {
            "total_steps": max_steps,
            "total_time_seconds": total_time,
            "steps_per_second": max_steps / total_time,
            "final_loss": final_metrics['loss'],
            "average_loss": avg_loss,
            "final_atoms": n_atoms,
            "average_atoms": aggregate_metrics['avg_atoms'],
            "max_atoms": aggregate_metrics['max_atoms'],
            "final_aggregates": final_metrics['n_aggregates'],
            "average_aggregates": avg_aggregates,
            "final_abstractions": final_metrics['n_abstractions'],
            "average_abstractions": avg_abstractions,
            "consolidation_count": final_metrics['consolidation_count'],
        },
        "efficiency": efficiency,
        "generations": generations,
        "model_summary": model.get_shared_params_summary(),
    }

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "tiny_shakespeare_results.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Dataset: Tiny Shakespeare (excerpt)")
    print(f"Training steps: {max_steps}")
    print(f"Final loss: {final_metrics['loss']:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Final atoms: {n_atoms}")
    print(f"Average atoms: {aggregate_metrics['avg_atoms']:.1f}")
    print(f"Max atoms: {aggregate_metrics['max_atoms']}")
    print(f"Final aggregates: {final_metrics['n_aggregates']}")
    print(f"Average aggregates: {avg_aggregates:.1f}")
    print(f"Final abstractions: {final_metrics['n_abstractions']}")
    print(f"Average abstractions: {avg_abstractions:.1f}")
    print(f"Consolidations: {final_metrics['consolidation_count']}")
    print(f"Total parameters: {n_params:,}")
    print(f"Atoms per parameter: {efficiency['atoms_per_parameter']:.4f}")
    print(f"Training time: {total_time:.1f}s ({max_steps/total_time:.2f} steps/sec)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
