import cv2
import os
import csv
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import threading
import queue
import time
import gc
from functools import wraps

class AsyncVideoAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotation Tool - Async Version")
        self.root.geometry("1200x800")

        # State
        self.qa_data = []
        self.current_video_qa = []
        self.csv_file_path = ""
        self.rejected_csv_file_path = ""
        self.accepted_qa_data = []
        self.saved_prompts = []
        self.video_dir = ""
        self.base_directory = ""
        self.current_video_path = ""
        self.mode = "video"
        self.frames = []
        self.frame_index = 0
        self.cap = None
        self.playing = False
        
        # Threading control
        self.current_thread = None
        self.stop_current_operation = threading.Event()
        self.operation_lock = threading.Lock()
        
        # UI state tracking
        self.is_loading = False

        self.PROMPT_BUILDERS = {
            "1": self.build_llava,
            "2": self.build_llava,
            "3": self.build_qwen
        }
        self.FILE_SUFFIX = {
            "1": "videollama2",
            "2": "llava_next",
            "3": "qwen"
        }

        self.setup_ui()
        print("Application initialized successfully")

    def setup_ui(self):
        main_container = Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.left_panel = Frame(main_container, width=200, bg="lightgray", relief=RAISED, bd=2)
        self.left_panel.pack(side=LEFT, fill=Y, padx=(0, 5))
        self.left_panel.pack_propagate(False)
        
        self.center_panel = Frame(main_container, bg="gray", relief=RAISED, bd=2)
        self.center_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
        
        self.right_panel = Frame(main_container, width=200, bg="lightgray", relief=RAISED, bd=2)
        self.right_panel.pack(side=RIGHT, fill=Y, padx=(5, 0))
        self.right_panel.pack_propagate(False)
        
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()

        # Status bar
        self.status_var = StringVar(value="Ready")
        status_bar = Label(self.root, textvariable=self.status_var, relief=SUNKEN, anchor=W)
        status_bar.pack(side=BOTTOM, fill=X)

    def setup_left_panel(self):
        Label(self.left_panel, text="Video Analysis", font=("Arial", 12, "bold"), bg="lightgray").pack(pady=10)
        
        self.accept_btn = Button(self.left_panel, text="Accept", bg="lightgreen", 
                                font=("Arial", 10, "bold"), width=15, height=2, 
                                command=self.accept_annotation)
        self.accept_btn.pack(pady=10)
        
        self.reject_btn = Button(self.left_panel, text="Reject", bg="lightcoral", 
                                font=("Arial", 10, "bold"), width=15, height=2, 
                                command=self.reject_annotation)
        self.reject_btn.pack(pady=5)
        
        Label(self.left_panel, text="Base Directory:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.dir_var = StringVar()
        dir_entry = Entry(self.left_panel, textvariable=self.dir_var, width=25)
        dir_entry.pack(pady=5, padx=10)
        
        Button(self.left_panel, text="Browse Directory", command=self.browse_directory, width=18).pack(pady=5)
        
        Label(self.left_panel, text="Select Video:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        
        listbox_frame = Frame(self.left_panel)
        listbox_frame.pack(pady=5, padx=10, fill=BOTH, expand=True)
        
        self.video_listbox = Listbox(listbox_frame, height=8)
        scrollbar = Scrollbar(listbox_frame, orient=VERTICAL, command=self.video_listbox.yview)
        self.video_listbox.config(yscrollcommand=scrollbar.set)
        self.video_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.video_listbox.bind('<Double-Button-1>', self.on_video_select)
        
        self.load_btn = Button(self.left_panel, text="Load Selected", command=self.load_selected_video, width=18)
        self.load_btn.pack(pady=5)

    def setup_center_panel(self):
        top_controls = Frame(self.center_panel)
        top_controls.pack(pady=10)
        
        self.switch_btn = Button(top_controls, text="Switch to Frame Analysis", 
                                command=self.switch_mode, font=("Arial", 10))
        self.switch_btn.pack(side=LEFT, padx=10)
        
        self.current_video_var = StringVar(value="No video loaded")
        Label(top_controls, textvariable=self.current_video_var, font=("Arial", 10)).pack(side=LEFT, padx=20)
        
        self.frame_label = Label(self.center_panel, width=80, height=30, bg="darkgray", 
                                text="Load a video to begin", font=("Arial", 12))
        self.frame_label.pack(pady=10, expand=True, fill=BOTH)
        
        self.qa_listbox = Listbox(self.center_panel, selectmode=EXTENDED, width=100, height=8)
        self.qa_listbox.pack(pady=5, padx=10)
        
        # Frame controls (hidden by default)
        self.controls_frame = Frame(self.center_panel)
        Button(self.controls_frame, text="Previous", command=self.prev_frame, font=("Arial", 10)).pack(side=LEFT, padx=5)
        self.frame_number_var = StringVar(value="")
        Label(self.controls_frame, textvariable=self.frame_number_var, font=("Arial", 10)).pack(side=LEFT, padx=10)
        Button(self.controls_frame, text="Next", command=self.next_frame, font=("Arial", 10)).pack(side=LEFT, padx=5)
        
        # Video controls
        self.video_controls_frame = Frame(self.center_panel)
        self.video_controls_frame.pack(pady=10)
        Button(self.video_controls_frame, text="Play/Pause", command=self.toggle_playback, font=("Arial", 10)).pack(side=LEFT, padx=5)
        Button(self.video_controls_frame, text="Stop", command=self.stop_video, font=("Arial", 10)).pack(side=LEFT, padx=5)

    def setup_right_panel(self):
        Label(self.right_panel, text="CSV File Controls", font=("Arial", 12, "bold"), bg="lightgray").pack(pady=10)
        
        Button(self.right_panel, text="Create New CSV", bg="lightblue", font=("Arial", 9, "bold"), 
               width=18, height=2, command=self.create_csv).pack(pady=10)
        Button(self.right_panel, text="Load Existing CSV", bg="lightyellow", font=("Arial", 9, "bold"), 
               width=18, height=2, command=self.load_csv_async).pack(pady=5)
        
        Label(self.right_panel, text="Current CSV:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.csv_status_var = StringVar(value="None loaded")
        Label(self.right_panel, textvariable=self.csv_status_var, bg="lightgray", 
              font=("Arial", 8), wraplength=180).pack(pady=5, padx=10)
        
        Label(self.right_panel, text="Export Options:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        Button(self.right_panel, text="Save CSV", command=self.save_csv, width=18).pack(pady=5)
        Button(self.right_panel, text="Export Summary", command=self.export_summary, width=18).pack(pady=5)
        
        Label(self.right_panel, text="Statistics:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.stats_frame = Frame(self.right_panel, bg="white", relief=SUNKEN, bd=1)
        self.stats_frame.pack(pady=5, padx=10, fill=X)
        
        stats_header = Frame(self.stats_frame, bg="lightblue")
        stats_header.pack(fill=X, pady=2)
        Label(stats_header, text="Status", font=("Arial", 8, "bold"), bg="lightblue", width=10).pack(side=LEFT, padx=2)
        Label(stats_header, text="Count", font=("Arial", 8, "bold"), bg="lightblue", width=8).pack(side=LEFT, padx=2)
        
        accepted_row = Frame(self.stats_frame, bg="white")
        accepted_row.pack(fill=X, pady=1)
        Label(accepted_row, text="Accepted", font=("Arial", 8), bg="white", width=10, anchor=W).pack(side=LEFT, padx=2)
        self.accepted_var = StringVar(value="0")
        Label(accepted_row, textvariable=self.accepted_var, font=("Arial", 8), bg="white", width=8, anchor=E).pack(side=LEFT, padx=2)
        
        rejected_row = Frame(self.stats_frame, bg="white")
        rejected_row.pack(fill=X, pady=1)
        Label(rejected_row, text="Rejected", font=("Arial", 8), bg="white", width=10, anchor=W).pack(side=LEFT, padx=2)
        self.rejected_var = StringVar(value="0")
        Label(rejected_row, textvariable=self.rejected_var, font=("Arial", 8), bg="white", width=8, anchor=E).pack(side=LEFT, padx=2)
        
        Label(self.right_panel, text="Chat Template:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.template_var = StringVar(value="1")
        self.template_values = {"VideoLLama2": "1", "Llava-Next-Video": "2", "Qwen-VL2-7b-hf": "3", "All": "4"}
        
        for (text, val) in self.template_values.items():
            Radiobutton(self.right_panel, text=text, variable=self.template_var, 
                       value=val, bg="lightgray").pack(anchor=W, padx=20)
        
        Button(self.right_panel, text="Save Template Selection", 
               command=self.save_chat_template_selection).pack(pady=10)
        Button(self.right_panel, text="Finish and Export", 
               command=self.finish_and_export_chat_template).pack(pady=5)

    # Utility methods for thread management
    def _display_bgr_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        display_width, display_height = 600, 400
        img_width, img_height = img.size
        ratio = min(display_width / img_width, display_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        self.frame_label.config(image=img_tk, text="")
        self.frame_label.image = img_tk
    
    def stop_current_thread(self):
        """Stop any currently running thread operation"""
        self.stop_current_operation.set()
        if self.current_thread and self.current_thread.is_alive():
            print("Stopping current operation...")
            # Wait a moment for thread to stop gracefully
            self.current_thread.join(timeout=1.0)

    def safe_thread_operation(func):
        """Decorator to safely handle thread operations"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.operation_lock:
                if self.is_loading:
                    print("Already loading, ignoring request")
                    return
                self.is_loading = True
                
            try:
                return func(self, *args, **kwargs)
            finally:
                self.is_loading = False
        return wrapper

    def cleanup_resources(self):
        """Clean up all video resources"""
        print("Cleaning up resources...")
        self.stop_video()
        
        # Clear frames
        if self.frames:
            try:
                self.frames.clear()
                gc.collect()
                print("Cleared frames from memory")
            except Exception as e:
                print(f"Error clearing frames: {e}")

    # Video Loading Methods
    def on_video_select(self, event):
        self.load_selected_video()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.base_directory = directory
            self.video_dir = directory
            self.dir_var.set(directory)
            self.populate_video_list()

    def populate_video_list(self):
        """Populate video list - this is fast so can be synchronous"""
        self.video_listbox.delete(0, END)
        if not self.base_directory:
            return
            
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        videos = []
        
        try:
            for file in os.listdir(self.base_directory):
                if file.lower().endswith(video_extensions):
                    videos.append(file)
            
            for video in videos:
                self.video_listbox.insert(END, video)
                
            self.status_var.set(f"Found {len(videos)} videos")
            print(f"Found {len(videos)} video files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not read directory: {str(e)}")

    @safe_thread_operation
    def load_selected_video(self):
        """Load selected video with proper thread management"""
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a video file.")
            return

        # Stop any current operation
        self.stop_current_thread()
        self.cleanup_resources()
        
        video_file = self.video_listbox.get(selection[0])
        video_path = os.path.join(self.base_directory, video_file)
        
        print(f"Loading video: {video_file}")
        self.current_video_path = video_path
        
        # Reset UI immediately
        self.frame_label.config(image='', text='Loading...')
        self.current_video_var.set(f"Loading: {video_file}")
        self.status_var.set("Loading video...")
        self.qa_listbox.delete(0, END)
        self.current_video_qa = []
        self.accepted_var.set("0")
        self.rejected_var.set("0")
        
        # Disable UI during loading
        self.load_btn.config(state=DISABLED)
        self.switch_btn.config(state=DISABLED)
        
        # Clear the stop event for new operation
        self.stop_current_operation.clear()
        
        # Start loading in separate thread
        if self.mode == "frame":
            self.current_thread = threading.Thread(
                target=self.load_video_frames_thread, 
                args=(video_path, video_file),
                daemon=True
            )
        else:
            self.current_thread = threading.Thread(
                target=self.load_video_playback_thread, 
                args=(video_path, video_file),
                daemon=True
            )
        
        self.current_thread.start()
        print(f"Started loading thread for {video_file}")

    def load_video_frames_thread(self, video_path, video_file):
        """Load video frames in background thread"""
        print(f"Starting frame loading thread for {video_file}")
        
        try:
            # Update status
            self.root.after(0, lambda: self.status_var.set("Opening video file..."))
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames to load: {total_frames}")
            
            self.root.after(0, lambda: self.status_var.set(f"Loading {total_frames} frames..."))
            
            frames = []
            frame_count = 0
            
            while True:
                # Check if we should stop
                if self.stop_current_operation.is_set():
                    print("Frame loading stopped by user")
                    cap.release()
                    return
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                frames.append(img)
                frame_count += 1
                
                # Update progress every 50 frames
                if frame_count % 50 == 0:
                    progress = f"Loaded {frame_count}/{total_frames} frames"
                    self.root.after(0, lambda p=progress: self.status_var.set(p))
                    print(progress)
            
            cap.release()
            
            if not frames:
                raise Exception("No frames were extracted from the video")
            
            print(f"Successfully loaded {len(frames)} frames")
            
            # Update UI on main thread
            def update_ui():
                if not self.stop_current_operation.is_set():
                    self.frames = frames
                    self.frame_index = 0
                    self.show_frame(0)
                    self.current_video_var.set(f"Current: {video_file}")
                    self.status_var.set(f"Loaded {len(frames)} frames")
                    self.show_qa_for_current_video()
                    print(f"UI updated with {len(frames)} frames")
                
                # Re-enable UI
                self.load_btn.config(state=NORMAL)
                self.switch_btn.config(state=NORMAL)
            
            self.root.after(0, update_ui)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading frames: {error_msg}")
            
            def show_error():
                if not self.stop_current_operation.is_set():
                    messagebox.showerror("Error", f"Error loading video frames: {error_msg}")
                    self.frame_label.config(image='', text="Failed to load video")
                    self.status_var.set("Error loading video")
                
                # Re-enable UI
                self.load_btn.config(state=NORMAL)
                self.switch_btn.config(state=NORMAL)
            
            self.root.after(0, show_error)

    def load_video_playback_thread(self, video_path, video_file):
        """Load video for playback in background thread - FIXED VERSION"""
        print(f"Starting playback loading thread for {video_file}")
        
        try:
            self.root.after(0, lambda: self.status_var.set("Initializing video playback..."))
            
            # Test if video can be opened
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                test_cap.release()
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_cap.release()
            
            print(f"Video info - FPS: {fps}, Frames: {frame_count}")

            def update_ui():
                if not self.stop_current_operation.is_set():
                    self.cap = cv2.VideoCapture(video_path)
                    if self.cap.isOpened():
                        self.current_video_var.set(f"Current: {video_file}")
                        self.status_var.set(f"Video ready - {frame_count} frames @ {fps:.1f} FPS")
                        self.show_qa_for_current_video()
                        
                        # Show first frame immediately
                        ret, frame = self.cap.read()
                        if ret:
                            self._display_bgr_frame(frame)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start for playback
                        print("Video playback initialized successfully")
                    else:
                        raise Exception("Failed to initialize video capture")
                
                # Re-enable UI
                self.load_btn.config(state=NORMAL)
                self.switch_btn.config(state=NORMAL)

            self.root.after(0, update_ui)

        except Exception as e:
            error_msg = str(e)
            print(f"Error initializing playback: {error_msg}")
            
            def show_error():
                if not self.stop_current_operation.is_set():
                    messagebox.showerror("Error", f"Error loading video: {error_msg}")
                    self.frame_label.config(image='', text="Failed to load video")
                    self.status_var.set("Error loading video")
                
                # Re-enable UI  
                self.load_btn.config(state=NORMAL)
                self.switch_btn.config(state=NORMAL)
            
            self.root.after(0, show_error)

    # Frame Display Methods
    def show_frame(self, idx):
        """Display frame with error handling"""
        if not self.frames or idx < 0 or idx >= len(self.frames):
            return
            
        try:
            self.frame_index = idx
            img = self.frames[self.frame_index]
            
            # Resize for display
            display_width = 600
            display_height = 400
            img_width, img_height = img.size
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            
            self.frame_label.config(image=img_tk, text="")
            self.frame_label.image = img_tk  # Keep reference
            self.frame_number_var.set(f"Frame {self.frame_index + 1} of {len(self.frames)}")
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
            self.frame_label.config(image='', text="Error displaying frame")

    def next_frame(self):
        if self.frames and not self.is_loading:
            next_idx = (self.frame_index + 1) % len(self.frames)
            self.show_frame(next_idx)

    def prev_frame(self):
        if self.frames and not self.is_loading:
            prev_idx = (self.frame_index - 1) % len(self.frames)
            self.show_frame(prev_idx)

    # Video Playback Methods
    def update_video(self):
        """Update video playback"""
        if not self.playing or self.cap is None:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize for display
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
                
                # Schedule next frame
                self.root.after(33, self.update_video)  # ~30 FPS
            else:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(33, self.update_video)
                
        except Exception as e:
            print(f"Error in video playback: {e}")
            self.stop_video()

    def toggle_playback(self):
        if self.cap is None or self.is_loading:
            return
            
        self.playing = not self.playing
        if self.playing:
            self.status_var.set("Playing video...")
            self.update_video()
        else:
            self.status_var.set("Video paused")

    def stop_video(self):
        """Stop video playback"""
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def switch_mode(self):
        """Switch between video and frame analysis modes"""
        if self.is_loading:
            messagebox.showinfo("Please Wait", "Cannot switch modes while loading.")
            return
            
        if self.mode == "video":
            self.mode = "frame"
            self.switch_btn.config(text="Switch to Video Playback")
            self.video_controls_frame.pack_forget()
            self.controls_frame.pack(pady=10)
            print("Switched to frame mode")
        else:
            self.mode = "video"
            self.switch_btn.config(text="Switch to Frame Analysis")
            self.controls_frame.pack_forget()
            self.video_controls_frame.pack(pady=10)
            print("Switched to video mode")
        
        # Reload current video in new mode if available
        if self.current_video_path:
            self.load_selected_video()

    # CSV Methods
    def load_csv_async(self):
        """Load CSV file asynchronously"""
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            return
            
        self.csv_status_var.set("Loading CSV...")
        
        def load_csv_thread():
            try:
                qa_data = []
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or "question" not in reader.fieldnames:
                        raise ValueError("Missing header or required columns in CSV.")
                    for row in reader:
                        qa_data.append(row)
                
                def update_ui():
                    rejected_csv_file_path = os.path.join(os.path.dirname(csv_path), 
                                                        "rejected_" + os.path.basename(csv_path))
                    self.csv_file_path = csv_path
                    self.rejected_csv_file_path = rejected_csv_file_path
                    self.qa_data = qa_data
                    
                    if len(qa_data) == 0:
                        self.csv_status_var.set("Loaded but empty")
                    else:
                        self.csv_status_var.set(f"Loaded: {os.path.basename(csv_path)}")
                        
                    self.show_qa_for_current_video()
                    self.status_var.set(f"CSV loaded: {len(qa_data)} entries")
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    self.csv_status_var.set("Failed to load CSV")
                    messagebox.showerror("Error", f"Could not load CSV file: {str(e)}")
                
                self.root.after(0, show_error)
        
        threading.Thread(target=load_csv_thread, daemon=True).start()

    def show_qa_for_current_video(self):
        """Show Q&A for current video"""
        self.qa_listbox.delete(0, END)
        self.current_video_qa = []
        
        if not self.current_video_path or not self.qa_data:
            self.accepted_var.set("0")
            self.rejected_var.set("0")
            return
            
        video_file_name = os.path.basename(self.current_video_path)
        
        for qa in self.qa_data:
            video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                         qa.get("video_path") or qa.get("file_name") or "")
            if video_file == video_file_name:
                self.current_video_qa.append(qa)
                category = qa.get("category", "Unknown")
                question = qa.get("question", "No question")
                answer = qa.get("answer", "No answer")
                qtext = f'[{category}] {question} (Ans: {answer})'
                self.qa_listbox.insert(END, qtext)
        
        # Count rejected items
        self.count_rejected_items(video_file_name)

    def count_rejected_items(self, video_file_name):
        """Count rejected items"""
        n_rejected = 0
        if self.rejected_csv_file_path and os.path.exists(self.rejected_csv_file_path):
            try:
                with open(self.rejected_csv_file_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        video_file = (row.get("video_file_path") or row.get("video_file") or 
                                    row.get("video_path") or row.get("file_name") or "")
                        if video_file == video_file_name:
                            n_rejected += 1
            except Exception as e:
                print(f"Error reading rejected CSV: {e}")
        
        self.rejected_var.set(str(n_rejected))
        self.accepted_var.set("0")

    def accept_annotation(self):
        """Accept selected annotations"""
        selected = self.qa_listbox.curselection()
        if not selected:
            messagebox.showinfo("Accept", "No question selected.")
            return
            
        newly_accepted = []
        for i in selected:
            if i < len(self.current_video_qa):
                qa = self.current_video_qa[i]
                if qa not in self.accepted_qa_data:
                    self.accepted_qa_data.append(qa)
                    newly_accepted.append(qa)
        
        self.accepted_var.set(str(int(self.accepted_var.get()) + len(newly_accepted)))
        
        if newly_accepted:
            messagebox.showinfo("Accept", 
                              f"Accepted: {len(newly_accepted)} new questions\n"
                              f"Total accepted: {len(self.accepted_qa_data)}")
        else:
            messagebox.showinfo("Accept", 
                              f"Selected questions were already accepted.\n"
                              f"Total accepted: {len(self.accepted_qa_data)}")
        
        self.status_var.set(f"Total accepted: {len(self.accepted_qa_data)}")

    def reject_annotation(self):
        """Reject selected annotations"""
        selected = self.qa_listbox.curselection()
        if not selected:
            messagebox.showinfo("Reject", "No question selected.")
            return
            
        to_reject = [self.current_video_qa[i] for i in selected if i < len(self.current_video_qa)]
        
        if not to_reject:
            return
            
        # Save rejections in background
        def save_rejections():
            try:
                self.append_qa_to_csv(self.rejected_csv_file_path, to_reject)
                
                def update_ui():
                    current_rejected = int(self.rejected_var.get())
                    self.rejected_var.set(str(current_rejected + len(selected)))
                    messagebox.showinfo("Reject", f"Rejected: {len(to_reject)} questions")
                    self.status_var.set(f"Rejected {len(to_reject)} items")
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save rejections: {str(e)}"))
        
        threading.Thread(target=save_rejections, daemon=True).start()

    def write_qa_to_csv(self, path, qa_list):
        """Write QA data to CSV file"""
        fieldnames = ["index", "video_file_path", "question", "category", "answer"]
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qa in qa_list:
                clean_row = {k: qa.get(k, "") for k in fieldnames}
                writer.writerow(clean_row)

    def append_qa_to_csv(self, path, qa_list):
        """Append QA data to CSV file"""
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

    # Prompt Builder Methods
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
        """Save chat template selection"""
        if not self.current_video_qa:
            messagebox.showinfo("Save", "No QA visible for this video.")
            return
            
        current_video_accepted = [
            qa for qa in self.current_video_qa 
            if qa in self.accepted_qa_data
        ]
        
        if not current_video_accepted:
            messagebox.showwarning("No Accepted Rows", 
                                 "No accepted Q&A pairs found for the current video.\n"
                                 "Please accept some questions first.")
            return
            
        choice = self.template_var.get()
        templates = ["1", "2", "3"] if choice == "4" else [choice]
        
        def generate_templates():
            try:
                added = 0
                for qa in current_video_accepted:
                    vpath = self.current_video_path
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    
                    if q and a:
                        for t in templates:
                            obj = self.PROMPT_BUILDERS[t](vpath, q, a, num_frames=4)
                            self.saved_prompts.append((t, obj))
                            added += 1
                
                def update_ui():
                    current_video_name = os.path.basename(self.current_video_path) if self.current_video_path else "current video"
                    template_names = {
                        "1": "VideoLLaMA2",
                        "2": "Llava-Next-Video", 
                        "3": "Qwen-VL2-7b-hf",
                        "4": "All templates"
                    }
                    
                    messagebox.showinfo(
                        "Save Template Selection",
                        f"Stored {added} prompt(s) from {len(current_video_accepted)} accepted Q&A pairs "
                        f"for {current_video_name} using {template_names[choice]} template(s) in memory.\n\n"
                        f"Total prompts in memory: {len(self.saved_prompts)}\n\n"
                        "Use 'Finish and Export' to write all saved prompts to disk, or continue selecting more videos."
                    )
                    self.status_var.set(f"Templates generated: {added}")
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate templates: {str(e)}"))
        
        self.status_var.set("Generating templates...")
        threading.Thread(target=generate_templates, daemon=True).start()

    def finish_and_export_chat_template(self):
        """Export chat templates"""
        if not self.accepted_qa_data:
            messagebox.showwarning("No Accepted Rows", 
                                 "No accepted Q&A pairs found.\n"
                                 "Please accept some questions first before exporting.")
            return
            
        template_choice = self.template_var.get()
        if template_choice not in ("1", "2", "3", "4"):
            messagebox.showerror("Invalid Choice", "Please enter 1, 2, 3, or 4 to select a valid template.")
            return
            
        out_dir = filedialog.askdirectory(title="Pick export directory")
        if not out_dir:
            return
        
        def export_templates():
            try:
                self.root.after(0, lambda: self.status_var.set("Exporting templates..."))
                
                prompts = []
                processed_count = 0
                skipped_count = 0
                
                for qa in self.accepted_qa_data:
                    video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                                qa.get("video_path") or qa.get("file_name") or "")
                    vpath = os.path.join(self.video_dir, video_file) if video_file else ""
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    
                    if vpath and q and a:
                        if template_choice == "4":
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
                
                if not prompts:
                    raise Exception("No valid prompts to export")
                
                # Write files
                if template_choice == "4":
                    template_groups = {"1": [], "2": [], "3": []}
                    for prompt in prompts:
                        template_groups[prompt["template"]].append(prompt["data"])
                    
                    files_written = []
                    for template_id, template_prompts in template_groups.items():
                        if template_prompts:
                            fname = f"model_train_{self.FILE_SUFFIX[template_id]}.jsonl"
                            fpath = os.path.join(out_dir, fname)
                            
                            with open(fpath, "w", encoding="utf-8") as fout:
                                for obj in template_prompts:
                                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            files_written.append(fname)
                    
                    def update_ui():
                        messagebox.showinfo(
                            "Export Complete",
                            f"All Templates Export Completed!\n\n"
                            f"Processed: {processed_count}\n"
                            f"Skipped: {skipped_count}\n"
                            f"Files exported: {', '.join(files_written)}"
                        )
                        self.status_var.set("Export completed")
                    
                    self.root.after(0, update_ui)
                    
                else:
                    fname = f"model_train_{self.FILE_SUFFIX[template_choice]}.jsonl"
                    fpath = os.path.join(out_dir, fname)
                    
                    with open(fpath, "w", encoding="utf-8") as fout:
                        for prompt in prompts:
                            fout.write(json.dumps(prompt["data"], ensure_ascii=False) + "\n")
                    
                    def update_ui():
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
                        self.status_var.set("Export completed")
                    
                    self.root.after(0, update_ui)
                
                # Clear saved prompts
                if self.saved_prompts:
                    self.saved_prompts.clear()
                    print("Cleared saved prompts from memory after export")
                    
            except Exception as e:
                def show_error():
                    messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
                    self.status_var.set("Export failed")
                
                self.root.after(0, show_error)
        
        threading.Thread(target=export_templates, daemon=True).start()

    def cleanup_on_exit(self):
        """Cleanup when application exits"""
        print("Application shutting down...")
        self.stop_current_thread()
        self.cleanup_resources()


if __name__ == "__main__":
    root = Tk()
    app = AsyncVideoAnnotationTool(root)
    
    # Handle window close event
    def on_closing():
        print("Closing application...")
        app.cleanup_on_exit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
        on_closing()