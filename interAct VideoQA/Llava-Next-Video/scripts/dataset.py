class LlavaVideoQADataset(Dataset):
    """
    A dataset class for video-based question answering.

    Expects a CSV with columns:
      - Video_File_Path: Path or filename of the video.
      - Questions: The question text.
      - Answers: The ground truth answer.

    This class samples frames from the video via PyAV and constructs a prompt in the format:

      USER: <question>
      <video_token_str>
      ASSISTANT:

    where <video_token_str> is the token expected by the model (derived from model.config.video_token_index).
    The ground truth answer is returned for evaluation.
    """
    def __init__(self, csv_file: str, video_dir: str, processor: any, model, num_frames: int = 8, extension: str = ".mp4"):
        self.data = pd.read_csv(csv_file, dtype={"Questions": str, "Answers": str})
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.extension = extension
        # Retrieve the expected video token string from the model configuration.
        self.video_token_str = processor.tokenizer.decode([model.config.video_token_index]).strip()

    def __len__(self):
        return len(self.data)

    def read_video_pyav(self, video_path: str, indices: np.ndarray) -> np.ndarray:
        container = av.open(video_path)
        frames = []
        container.seek(0)
        start_idx = indices[0]
        end_idx = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_idx:
                break
            if i >= start_idx and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        video_file = row["Video_File_Path"].strip()
        question = row["Questions"].strip()
        answer = row["Answers"].strip()
        video_path = os.path.join(self.video_dir, video_file)
        if not os.path.splitext(video_path)[1]:
            video_path += self.extension
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
        clip = self.read_video_pyav(video_path, indices)
        # Construct the prompt using the expected video token.
        prompt = f"USER: {question}\n{self.video_token_str}\nASSISTANT:"
        inputs = self.processor(text=prompt, videos=clip, padding=True, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if "pixel_values" in inputs and "pixel_values_videos" not in inputs:
            inputs["pixel_values_videos"] = inputs.pop("pixel_values")
        inputs["ground_truth"] = answer
        return inputs
