
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(">>> Pre-training Evaluation")
model.eval()
pretrain_predictions = []
with torch.no_grad():
    for j, batch in enumerate(tqdm(eval_loader, desc="Pre-training Eval")):
        if j >= 20:
            break
        gen_kwargs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        generated_ids = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False)
        prompt_length = batch["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_length:]
        predictions = processor.batch_decode(new_tokens, skip_special_tokens=True)
        pretrain_predictions.append((predictions, batch.get("ground_truth", ["N/A"])))
print(">>> Pre-training Evaluation Predictions:")
for preds, gt in pretrain_predictions:
    print("Prediction:", preds)
    print("Ground Truth:", gt)

