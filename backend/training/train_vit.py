import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for Deepfake Detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset with 'train' and 'val' subfolders (each containing 'real' and 'fake' folders)")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_vit", help="Where to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()

    # We start from the model we are currently using in TrustLayer, or a base ViT
    model_name = "dima806/deepfake_vs_real_image_detection"
    
    print(f"Loading image processor and model: {model_name}...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True 
    )

    print(f"Loading dataset from {args.data_dir}...")
    # Requires folder structure: data_dir/train/real, data_dir/train/fake
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)

    def transforms(examples):
        # Apply the image processor to the images
        examples["pixel_values"] = [
            image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            for image in examples["image"]
        ]
        # Trainer expects 'labels' (not 'label') to compute cross-entropy loss
        examples["labels"] = examples["label"]
        return examples

    print("Applying image transformations...")
    dataset = dataset.map(transforms, batched=True, remove_columns=["image", "label"])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # You might want to adjust evaluation_strategy to 'epoch' if you are using transformers < 4.41.0
    # In newer versions it's 'eval_strategy'.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        warmup_steps=50,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["train"],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print("Done! You can now load this model in TrustLayer by updating models/loader.py")

if __name__ == "__main__":
    main()
