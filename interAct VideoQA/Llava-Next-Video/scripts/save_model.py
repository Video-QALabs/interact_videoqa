save_dir = "<Change Dir>"  # update this path as needed
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print("Model and processor saved to:", save_dir)