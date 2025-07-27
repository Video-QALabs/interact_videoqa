import cv2
import os
import csv
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import pandas as pd
import threading

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
        self.accepted_qa_data = []  # Track accepted Q&A pairs
        
        # Initialize missing variables
        self.saved_prompts = []
        self.video_dir = ""

        # Prompt builders dictionary
        self.PROMPT_BUILDERS = {
            "1": self.build_llava,   # Video-LLaMA-2 / -3
            "2": self.build_llava,   # Llava-NEXT VIDEO 
            "3": self.build_qwen     # Qwen-VL-2-7B-HF
        }
        
        # File suffix for export
        self.FILE_SUFFIX = {
            "1": "videollama2", 
            "2": "llava_next", 
            "3": "qwen"
        }

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

        # Create frame for listbox with scrollbar
        listbox_frame = Frame(self.left_panel)
        listbox_frame.pack(pady=5, padx=10, fill=BOTH, expand=True)

        # Create listbox with scrollbar
        self.video_listbox = Listbox(listbox_frame, height=8)
        scrollbar = Scrollbar(listbox_frame, orient=VERTICAL, command=self.video_listbox.yview)
        self.video_listbox.config(yscrollcommand=scrollbar.set)
        
        # Pack listbox and scrollbar
        self.video_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
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
        Button(self.right_panel, text="qa-generation", command=self.export_summary,
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

        # Chat Template Selection
        Label(self.right_panel, text="Chat Template:", bg="lightgray",
              font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.template_var = StringVar(value="1")
        self.template_values = {
            "VideoLLama2": "1",
            "Llava-Next-Video": "2",
            "Qwen-VL2-7b-hf": "3",
            "All": "4"
        }

        for (text, val) in self.template_values.items():
            Radiobutton(self.right_panel, text=text, variable=self.template_var, 
                        value=val, bg="lightgray").pack(anchor=W, padx=20)

        # Save button
        Button(self.right_panel, text="Save Template Selection", 
               command=self.save_chat_template_selection).pack(pady=10)

        # Export button
        Button(self.right_panel, text="Finish and Export", 
               command=self.finish_and_export_chat_template).pack(pady=5)

    def switch_mode(self):
        """Switch between video playback and frame-by-frame analysis modes."""
        if self.mode == "video":
            self.mode = "frame"
            self.switch_btn.config(text="Switch to Video Playback")
            self.video_controls_frame.pack_forget()
            self.controls_frame.pack(pady=10)
            
            # If we have a current video, load it as frames
            if self.current_video_path:
                self.stop_video()
                threading.Thread(target=self.load_video_frames_safe, 
                               args=(self.current_video_path,), daemon=True).start()
        else:
            self.mode = "video"
            self.switch_btn.config(text="Switch to Frame Analysis")
            self.controls_frame.pack_forget()
            self.video_controls_frame.pack(pady=10)
            
            # If we have a current video, start video playback
            if self.current_video_path:
                self.frames.clear()
                self.play_video(self.current_video_path)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.base_directory = directory
            self.video_dir = directory  # Set video_dir as well
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
        """Load and display the selected video with full state reset and threading."""
        self.stop_video()  # Stop any existing video playback
        self.frames.clear()
        self.frame_index = 0
        self.frame_label.config(image='', text='Loading...')
        self.current_video_var.set("No video loaded")
        self.qa_listbox.delete(0, END)
        self.current_video_qa = []
        self.accepted_var.set("0")
        self.rejected_var.set("0")
        self.root.update_idletasks()

        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a video file.")
            return

        video_file = self.video_listbox.get(selection[0])
        video_path = os.path.join(self.base_directory, video_file)
        self.current_video_path = video_path
        self.current_video_var.set(f"Current: {video_file}")

        self.show_qa_for_current_video()

        if self.mode == "frame":
            threading.Thread(target=self.load_video_frames_safe, args=(video_path,), daemon=True).start()
        else:
            self.play_video(video_path)

    def load_video_frames_safe(self, path):
        """Load video frames in a background thread and update UI safely."""
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video file."))
                return

            local_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                local_frames.append(img)
            cap.release()

            if not local_frames:
                self.root.after(0, lambda: messagebox.showerror("Error", "No frames extracted from video."))
                return

            def finish_loading():
                self.frames = local_frames
                self.show_frame(0)

            self.root.after(0, finish_loading)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error loading video: {e}"))

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
            try:
                self.csv_file_path = csv_path
                self.rejected_csv_file_path = os.path.join(
                    os.path.dirname(csv_path),
                    "rejected_" + os.path.basename(csv_path)
                )
                self.csv_status_var.set(f"Loaded: {os.path.basename(csv_path)}")
                self.qa_data = []
                
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Check what columns are available
                    fieldnames = reader.fieldnames
                    print(f"CSV columns found: {fieldnames}")
                    
                    for row in reader:
                        self.qa_data.append(row)
                
                print(f"Loaded {len(self.qa_data)} rows from CSV")
                if self.qa_data:
                    print(f"Sample row: {self.qa_data[0]}")
                
                self.show_qa_for_current_video()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load CSV file: {str(e)}")
                print(f"CSV loading error: {e}")

    def show_qa_for_current_video(self):
        self.qa_listbox.delete(0, END)
        self.current_video_qa = []
        if not self.current_video_path or not self.qa_data:
            self.accepted_var.set("0")
            self.rejected_var.set("0")
            return
        
        video_file_name = os.path.basename(self.current_video_path)
        
        for qa in self.qa_data:
            # Handle different possible column names for video file
            video_file = qa.get("video_file_path") or qa.get("video_file") or qa.get("video_path") or qa.get("file_name") or ""
            
            if video_file == video_file_name:
                self.current_video_qa.append(qa)
                # Handle different possible column names
                category = qa.get("category", "Unknown")
                question = qa.get("question", "No question")
                answer = qa.get("answer", "No answer")
                qtext = f'[{category}] {question} (Ans: {answer})'
                self.qa_listbox.insert(END, qtext)
        
        # Calculate number rejected by scanning the rejected file
        n_rejected = 0
        if os.path.exists(self.rejected_csv_file_path):
            try:
                with open(self.rejected_csv_file_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        video_file = row.get("video_file_path") or row.get("video_file") or row.get("video_path") or row.get("file_name") or ""
                        if video_file == video_file_name:
                            n_rejected += 1
            except Exception as e:
                print(f"Error reading rejected CSV: {e}")
        
        self.rejected_var.set(str(n_rejected))
        self.accepted_var.set("0") 

    def accept_annotation(self):
        selected = self.qa_listbox.curselection()
        if not selected:
            messagebox.showinfo("Accept", "No question selected.")
            return
        
        # Add selected Q&A pairs to accepted list
        newly_accepted = []
        for i in selected:
            qa = self.current_video_qa[i]
            # Avoid duplicates
            if qa not in self.accepted_qa_data:
                self.accepted_qa_data.append(qa)
                newly_accepted.append(qa)
        
        # Update counter
        self.accepted_var.set(str(int(self.accepted_var.get()) + len(newly_accepted)))
        
        if newly_accepted:
            messagebox.showinfo("Accept", f"Accepted: {len(newly_accepted)} new questions\nTotal accepted: {len(self.accepted_qa_data)}")
        else:
            messagebox.showinfo("Accept", f"Selected questions were already accepted.\nTotal accepted: {len(self.accepted_qa_data)}")
        
        print(f"Total accepted Q&A pairs: {len(self.accepted_qa_data)}")
       
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
         messagebox.showinfo("Export", "qa-generation functionality would go here")
    
    # ------------------ PROMPT BUILDER METHODS -------------------
    def build_qwen(self, video_path, question, answer, num_frames=10):
        return {
            "conversations": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video",
                     "video": {"video_path": video_path,
                               "fps": 1,
                               "max_frames": num_frames}},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
        }

    def build_llava(self, video_path, question, answer, num_frames=10):
        prompt = f"USER: {question}\n<|video|>\nASSISTANT:"
        return {
            "prompt": prompt,
            "answer": answer,
            "video_path": video_path
        }

    def build_llama2(self, video_path, question, answer, num_frames=10):
        return {
            "conversations": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": num_frames}},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer}
                ]}
            ]
        }

    def save_chat_template_selection(self):
        """Converts only the ACCEPTED Q&A currently visible in the GUI to the currently chosen template and stores them in memory."""
        if not self.current_video_qa:
            messagebox.showinfo("Save", "No QA visible for this video.")
            return

        # Filter current video Q&A for only accepted ones
        current_video_accepted = []
        for qa in self.current_video_qa:
            if qa in self.accepted_qa_data:
                current_video_accepted.append(qa)
        
        if not current_video_accepted:
            messagebox.showwarning("No Accepted Rows", "No accepted Q&A pairs found for the current video.\nPlease accept some questions first.")
            return

        choice = self.template_var.get()           # "1" / "2" / "3" / "4"
        templates = ["1", "2", "3"] if choice == "4" else [choice]

        added = 0
        for qa in current_video_accepted:
            vpath = self.current_video_path
            # Handle different possible column names
            q = qa.get("question", "")
            a = qa.get("answer", "")
            
            if q and a:  # Only process if we have valid question and answer
                for t in templates:
                    obj = self.PROMPT_BUILDERS[t](vpath, q, a, num_frames=4)
                    self.saved_prompts.append((t, obj))
                    added += 1

        current_video_name = os.path.basename(self.current_video_path) if self.current_video_path else "current video"
        template_names = {
            "1": "VideoLLaMA2",
            "2": "Llava-Next-Video", 
            "3": "Qwen-VL2-7b-hf",
            "4": "All templates"
        }
        
        messagebox.showinfo(
            "Save Template Selection",
            f"Stored {added} prompt(s) from {len(current_video_accepted)} accepted Q&A pairs for {current_video_name} using {template_names[choice]} template(s) in memory.\n\n"
            f"Total prompts in memory: {len(self.saved_prompts)}\n\n"
            "Use 'Finish and Export' to write all saved prompts to disk, or continue selecting more videos."
        )

    def finish_and_export_chat_template(self):
        """Exports only ACCEPTED Q&A pairs using a selected prompt template (out of 4 choices)."""
        if not self.accepted_qa_data:
            messagebox.showwarning("No Accepted Rows", "No accepted Q&A pairs found.\nPlease accept some questions first before exporting.")
            return

        # Ask user to select the template to export (1–4)
        template_choice = self.template_var.get()  
        if template_choice not in ("1", "2", "3", "4"):
            messagebox.showerror("Invalid Choice", "Please enter 1, 2, 3, or 4 to select a valid template.")
            return

        # Ask for output directory
        out_dir = filedialog.askdirectory(title="Pick export directory")
        if not out_dir:
            return

        prompts = []
        processed_count = 0
        skipped_count = 0

        for qa in self.accepted_qa_data:
            video_file = qa.get("video_file_path") or qa.get("video_file") or qa.get("video_path") or qa.get("file_name") or ""
            vpath = os.path.join(self.video_dir, video_file) if video_file else ""

            q = qa.get("question", "")
            a = qa.get("answer", "")

            if vpath and q and a:
                if template_choice == "4":  # All templates
                    for template_id in ["1", "2", "3"]:
                        prompts.append({
                            "template": template_id,
                            "data": self.PROMPT_BUILDERS[template_id](vpath, q, a, num_frames=4)
                        })
                else:
                    prompts.append({
                        "template": template_choice,
                        "data": self.PROMPT_BUILDERS[template_choice](vpath, q, a, num_frames=4)
                    })
                processed_count += 1
            else:
                skipped_count += 1

        # Write selected template file(s)
        if prompts:
            if template_choice == "4":  # Export all templates
                # Group by template
                template_groups = {"1": [], "2": [], "3": []}
                for prompt in prompts:
                    template_groups[prompt["template"]].append(prompt["data"])
                
                # Write separate files for each template
                files_written = []
                for template_id, template_prompts in template_groups.items():
                    if template_prompts:
                        fname = f"model_train_{self.FILE_SUFFIX[template_id]}.jsonl"
                        fpath = os.path.join(out_dir, fname)
                        try:
                            with open(fpath, "w", encoding="utf-8") as fout:
                                for obj in template_prompts:
                                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            files_written.append(fname)
                        except Exception as exc:
                            messagebox.showerror("Export Error", f"Failed to write file {fname}: {exc}")
                            return
                
                messagebox.showinfo(
                    "Export Complete",
                    f"All Templates Export Completed!\n\n"
                    f"Processed: {processed_count}\n"
                    f"Skipped: {skipped_count}\n"
                    f"Files exported: {', '.join(files_written)}"
                )
            else:
                # Single template export
                fname = f"model_train_{self.FILE_SUFFIX[template_choice]}.jsonl"
                fpath = os.path.join(out_dir, fname)
                try:
                    with open(fpath, "w", encoding="utf-8") as fout:
                        for prompt in prompts:
                            fout.write(json.dumps(prompt["data"], ensure_ascii=False) + "\n")
                    
                    template_names = {
                        "1": "VideoLLaMA2",
                        "2": "Llava-Next-Video", 
                        "3": "Qwen-VL2-7b-hf"
                    }
                    
                    messagebox.showinfo(
                        "Export Complete",
                        f"Accepted Q&A Export Completed!\n\n"
                        f"Template used: {template_names[template_choice]}\n"
                        f"Processed: {processed_count}\n"
                        f"Skipped: {skipped_count}\n"
                        f"Exported to: {fname}"
                    )
                except Exception as exc:
                    messagebox.showerror("Export Error", f"Failed to write file: {exc}")
                    return
        else:
            messagebox.showwarning("Export Failed", "No valid prompts to export for the selected template.")

        if self.saved_prompts:
            self.saved_prompts.clear()
            print("Cleared saved prompts from memory after export")



# Main application
if __name__ == "__main__":
    root = Tk()
    app = VideoAnnotationTool(root)
    root.mainloop()