import cv2
import os
import csv
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class VideoAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotation Tool")
        self.root.geometry("1200x800")

        # Global variables
        self.base_directory = ""
        self.current_video_path = ""
        self.mode = "video"  # "video" or "frame"
        self.frames = []
        self.frame_index = 0
        self.cap = None
        self.playing = False

        # CSV/QA state
        self.qa_data = []
        self.current_video_qa = []
        self.csv_file_path = ""
        self.rejected_csv_file_path = ""

        self.setup_ui()

    def setup_ui(self):
        # Main container with three sections
        main_container = Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Left panel - Accept/Reject controls
        self.left_panel = Frame(main_container, width=200, bg="lightgray", relief=RAISED, bd=2)
        self.left_panel.pack(side=LEFT, fill=Y, padx=(0, 5))
        self.left_panel.pack_propagate(False)

        # Center panel - Video display
        self.center_panel = Frame(main_container, bg="gray", relief=RAISED, bd=2)
        self.center_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        # Right panel - CSV file controls
        self.right_panel = Frame(main_container, width=200, bg="lightgray", relief=RAISED, bd=2)
        self.right_panel.pack(side=RIGHT, fill=Y, padx=(5, 0))
        self.right_panel.pack_propagate(False)

        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()

    def setup_left_panel(self):
        # Left panel title
        Label(self.left_panel, text="Video Analysis", font=("Arial", 12, "bold"),
              bg="lightgray").pack(pady=10)

        # Accept/Reject buttons
        self.accept_btn = Button(self.left_panel, text="Accept", bg="lightgreen",
                                 font=("Arial", 10, "bold"), width=15, height=2,
                                 command=self.accept_annotation)
        self.accept_btn.pack(pady=10)

        self.reject_btn = Button(self.left_panel, text="Reject", bg="lightcoral",
                                 font=("Arial", 10, "bold"), width=15, height=2,
                                 command=self.reject_annotation)
        self.reject_btn.pack(pady=5)

        # Directory selection
        Label(self.left_panel, text="Base Directory:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))

        self.dir_var = StringVar()
        dir_entry = Entry(self.left_panel, textvariable=self.dir_var, width=25)
        dir_entry.pack(pady=5, padx=10)

        Button(self.left_panel, text="Browse Directory", command=self.browse_directory,
               width=18).pack(pady=5)

        # Video file selection
        Label(self.left_panel, text="Select Video:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))

        self.video_listbox = Listbox(self.left_panel, height=8)
        self.video_listbox.pack(pady=5, padx=10, fill=X)
        self.video_listbox.bind('<Double-Button-1>', self.on_video_select)

        Button(self.left_panel, text="Load Selected", command=self.load_selected_video,
               width=18).pack(pady=5)

    def setup_center_panel(self):
        # Top controls
        top_controls = Frame(self.center_panel)
        top_controls.pack(pady=10)

        # Mode switch button
        self.switch_btn = Button(top_controls, text="Switch to Frame Analysis",
                                 command=self.switch_mode, font=("Arial", 10))
        self.switch_btn.pack(side=LEFT, padx=10)

        # Current video label
        self.current_video_var = StringVar(value="No video loaded")
        Label(top_controls, textvariable=self.current_video_var,
              font=("Arial", 10)).pack(side=LEFT, padx=20)

        # Video display area
        self.frame_label = Label(self.center_panel, width=80, height=30, bg="darkgray",
                                 text="Load a video to begin", font=("Arial", 12))
        self.frame_label.pack(pady=10, expand=True, fill=BOTH)

        # QA Listbox (selectable questions)
        self.qa_listbox = Listbox(self.center_panel, selectmode=EXTENDED, width=100, height=8)
        self.qa_listbox.pack(pady=5, padx=10)

        # Frame controls (initially hidden)
        self.controls_frame = Frame(self.center_panel)
        Button(self.controls_frame, text="Previous", command=self.prev_frame,
               font=("Arial", 10)).pack(side=LEFT, padx=5)
        self.frame_number_var = StringVar(value="")
        Label(self.controls_frame, textvariable=self.frame_number_var,
              font=("Arial", 10)).pack(side=LEFT, padx=10)
        Button(self.controls_frame, text="Next", command=self.next_frame,
               font=("Arial", 10)).pack(side=LEFT, padx=5)

        # Video playback controls
        self.video_controls_frame = Frame(self.center_panel)
        self.video_controls_frame.pack(pady=10)
        Button(self.video_controls_frame, text="Play/Pause", command=self.toggle_playback,
               font=("Arial", 10)).pack(side=LEFT, padx=5)
        Button(self.video_controls_frame, text="Stop", command=self.stop_video,
               font=("Arial", 10)).pack(side=LEFT, padx=5)

    def setup_right_panel(self):
        Label(self.right_panel, text="CSV File Controls", font=("Arial", 12, "bold"),
              bg="lightgray").pack(pady=10)

        Button(self.right_panel, text="Create New CSV", bg="lightblue",
               font=("Arial", 9, "bold"), width=18, height=2,
               command=self.create_csv).pack(pady=10)

        Button(self.right_panel, text="Load Existing CSV", bg="lightyellow",
               font=("Arial", 9, "bold"), width=18, height=2,
               command=self.load_csv).pack(pady=5)

        Label(self.right_panel, text="Current CSV:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.csv_status_var = StringVar(value="None loaded")
        Label(self.right_panel, textvariable=self.csv_status_var, bg="lightgray",
              font=("Arial", 8), wraplength=180).pack(pady=5, padx=10)

        Label(self.right_panel, text="Export Options:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))
        Button(self.right_panel, text="Save CSV", command=self.save_csv,
               width=18).pack(pady=5)
        Button(self.right_panel, text="Export Summary", command=self.export_summary,
               width=18).pack(pady=5)

        Label(self.right_panel, text="Statistics:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))

        self.stats_frame = Frame(self.right_panel, bg="white", relief=SUNKEN, bd=1)
        self.stats_frame.pack(pady=5, padx=10, fill=X)
        stats_header = Frame(self.stats_frame, bg="lightblue")
        stats_header.pack(fill=X, pady=2)
        Label(stats_header, text="Status", font=("Arial", 8, "bold"),
              bg="lightblue", width=10).pack(side=LEFT, padx=2)
        Label(stats_header, text="Count", font=("Arial", 8, "bold"),
              bg="lightblue", width=8).pack(side=LEFT, padx=2)

        # Accepted row
        accepted_row = Frame(self.stats_frame, bg="white")
        accepted_row.pack(fill=X, pady=1)
        Label(accepted_row, text="Accepted", font=("Arial", 8),
              bg="white", width=10, anchor=W).pack(side=LEFT, padx=2)
        self.accepted_var = StringVar(value="0")
        Label(accepted_row, textvariable=self.accepted_var, font=("Arial", 8),
              bg="white", width=8, anchor=E).pack(side=LEFT, padx=2)

        # Rejected row
        rejected_row = Frame(self.stats_frame, bg="white")
        rejected_row.pack(fill=X, pady=1)
        Label(rejected_row, text="Rejected", font=("Arial", 8),
              bg="white", width=10, anchor=W).pack(side=LEFT, padx=2)
        self.rejected_var = StringVar(value="0")
        Label(rejected_row, textvariable=self.rejected_var, font=("Arial", 8),
              bg="white", width=8, anchor=E).pack(side=LEFT, padx=2)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.base_directory = directory
            self.dir_var.set(directory)
            self.populate_video_list()

    def populate_video_list(self):
        self.video_listbox.delete(0, END)
        if not self.base_directory:
            return
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        try:
            for file in os.listdir(self.base_directory):
                if file.lower().endswith(video_extensions):
                    self.video_listbox.insert(END, file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read directory: {str(e)}")

    def on_video_select(self, event):
        self.load_selected_video()

    def load_selected_video(self):
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a video file.")
            return
        video_file = self.video_listbox.get(selection[0])
        video_path = os.path.join(self.base_directory, video_file)
        self.current_video_path = video_path
        self.current_video_var.set(f"Current: {video_file}")
        # Show QAs for current video
        self.show_qa_for_current_video()
        # Load video
        if self.mode == "frame":
            self.load_video_frames(video_path)
        else:
            self.play_video(video_path)

    def switch_mode(self):
        if self.mode == "video":
            self.mode = "frame"
            self.switch_btn.config(text="Switch to Video Analysis")
            self.stop_video()
            self.video_controls_frame.pack_forget()
            self.controls_frame.pack(pady=10)
            if self.current_video_path:
                self.load_video_frames(self.current_video_path)
        else:
            self.mode = "video"
            self.switch_btn.config(text="Switch to Frame Analysis")
            self.controls_frame.pack_forget()
            self.video_controls_frame.pack(pady=10)
            if self.current_video_path:
                self.play_video(self.current_video_path)

    def load_video_frames(self, path):
        self.frames.clear()
        self.frame_index = 0
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                return
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self.frames.append(img)
            cap.release()
            if self.frames:
                self.show_frame(0)
            else:
                messagebox.showerror("Error", "No frames extracted from video.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_frame(self, idx):
        if not self.frames:
            return
        self.frame_index = idx % len(self.frames)
        img = self.frames[self.frame_index]
        display_width = 600
        display_height = 400
        img_width, img_height = img.size
        ratio = min(display_width / img_width, display_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        self.frame_label.config(image=img_tk, text="")
        self.frame_label.image = img_tk
        self.frame_number_var.set(f"Frame {self.frame_index + 1} of {len(self.frames)}")

    def next_frame(self):
        if self.frames:
            self.show_frame((self.frame_index + 1) % len(self.frames))

    def prev_frame(self):
        if self.frames:
            self.show_frame((self.frame_index - 1) % len(self.frames))

    def play_video(self, path):
        self.stop_video()
        try:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                return
            self.playing = True
            self.update_video()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_video(self):
        if not self.playing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            display_width = 600
            display_height = 400
            img_width, img_height = img.size
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            self.frame_label.config(image=img_tk, text="")
            self.frame_label.image = img_tk
            self.root.after(30, self.update_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(30, self.update_video)

    def toggle_playback(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.update_video()

    def stop_video(self):
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------ QA CSV OPERATIONS -------------------
    def load_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if csv_path:
            self.csv_file_path = csv_path
            self.rejected_csv_file_path = os.path.join(
                os.path.dirname(csv_path),
                "rejected_" + os.path.basename(csv_path)
            )
            self.csv_status_var.set(f"Loaded: {os.path.basename(csv_path)}")
            self.qa_data = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.qa_data.append(row)
            self.show_qa_for_current_video()

    def show_qa_for_current_video(self):
        self.qa_listbox.delete(0, END)
        self.current_video_qa = []
        if not self.current_video_path or not self.qa_data:
            self.accepted_var.set("0")
            self.rejected_var.set("0")
            return
        video_file_name = os.path.basename(self.current_video_path)
        for qa in self.qa_data:
            if qa["video_file_path"] == video_file_name:
                self.current_video_qa.append(qa)
                qtext = f'[{qa["category"]}] {qa["question"]} (Ans: {qa["answer"]})'
                self.qa_listbox.insert(END, qtext)
        # Calculate number rejected by scanning the rejected file
        n_rejected = 0
        if os.path.exists(self.rejected_csv_file_path):
            with open(self.rejected_csv_file_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["video_file_path"] == video_file_name:
                        n_rejected += 1
        self.rejected_var.set(str(n_rejected))
        self.accepted_var.set("0") 


    def accept_annotation(self):
        selected = self.qa_listbox.curselection()
        if not selected:
            messagebox.showinfo("Accept", "No question selected.")
            return
        # Just increment counter, do NOT remove anything
        self.accepted_var.set(str(int(self.accepted_var.get()) + len(selected)))
        messagebox.showinfo("Accept", f"Accepted: {len(selected)} questions")
        # No file changes!
       
    def reject_annotation(self):
        selected = self.qa_listbox.curselection()
        if not selected:
            messagebox.showinfo("Reject", "No question selected.")
            return
        to_reject = [self.current_video_qa[i] for i in selected]
        # Add to rejected CSV ONLY, do NOT remove from self.qa_data or main CSV
        self.append_qa_to_csv(self.rejected_csv_file_path, to_reject)
        self.rejected_var.set(str(int(self.rejected_var.get()) + len(selected)))
        messagebox.showinfo("Reject", f"Rejected: {len(to_reject)} questions")


    def write_qa_to_csv(self, path, qa_list):
        fieldnames = ["index", "video_file_path", "question", "category", "answer"]
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qa in qa_list:
                # Only keep the right fields
                clean_row = {k: qa.get(k, "") for k in fieldnames}
                writer.writerow(clean_row)

    def append_qa_to_csv(self, path, qa_list):
        if not qa_list:
            return
        fieldnames = ["index", "video_file_path", "question", "category", "answer"]
        write_header = not os.path.exists(path)
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for qa in qa_list:
                clean_row = {k: qa.get(k, "") for k in fieldnames}
                writer.writerow(clean_row)

    def create_csv(self):
        messagebox.showinfo("CSV", "Create CSV functionality would go here")

    def save_csv(self):
        messagebox.showinfo("CSV", "Save CSV functionality would go here")

    def export_summary(self):
        messagebox.showinfo("Export", "Export summary functionality would go here")

# Main application
if __name__ == "__main__":
    root = Tk()
    app = VideoAnnotationTool(root)
    root.mainloop()
