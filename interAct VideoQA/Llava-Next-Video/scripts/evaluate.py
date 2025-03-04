
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
num_epochs = config["max_epochs"]
accumulation_steps = config["accumulate_grad_batches"]
max_train_batches = 356 # limit each epoch to 100 batches
max_eval_batches = 20    # evaluate only 20 batches

print("\n>>> Starting Finetuning...")
model.train()
for epoch in range(num_epochs):
    print(f"\n=== Starting Epoch {epoch+1}/{num_epochs} ===")
    epoch_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        if (i + 1) > max_train_batches:
            break
        train_inputs = {k: v for k, v in batch.items() if k != "ground_truth"}
        train_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in train_inputs.items()}
        outputs = model(**train_inputs, labels=batch["input_ids"].to(device))
        loss = outputs.loss / accumulation_steps
        loss.backward()
        epoch_loss += loss.item() * accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                gen_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
                gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_inputs.items()}
                generated_ids = model.generate(**gen_inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
                prompt_length = batch["input_ids"].shape[1]
                new_tokens = generated_ids[:, prompt_length:]
                predictions = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_prompt = processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                print(f"\n[Debug] Batch {i+1}")
                print("Decoded Prompt:", decoded_prompt)
                print("Prompt Length:", prompt_length)
                full_generated = processor.batch_decode(generated_ids, skip_special_tokens=True)
                print("Full Generated Output:", full_generated)
                print("Predictions:", predictions)
            model.train()
    avg_loss = epoch_loss / min(len(train_loader), max_train_batches)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}")
    torch.cuda.empty_cache()

    print(">>> Starting Evaluation")
    model.eval()
    all_predictions = []
    eval_batches = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            if eval_batches >= max_eval_batches:
                break
            gen_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
            gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_inputs.items()}
            generated_ids = model.generate(**gen_inputs, max_new_tokens=128, do_sample=False)
            prompt_length = batch["input_ids"].shape[1]
            new_tokens = generated_ids[:, prompt_length:]
            predictions = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ground_truth = batch.get("ground_truth", ["N/A"])
            all_predictions.append((predictions, ground_truth))
            eval_batches += 1
    print(">>> Evaluation Predictions:")
    for preds, gt in all_predictions:
        print("Prediction:", preds)
        print("Ground Truth:", gt)
    # model.train()
