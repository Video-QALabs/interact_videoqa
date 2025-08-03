import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import Counter
import re
import csv

from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import threading
import queue
import numpy as np
import time
import gc
from ultralytics import YOLO
from functools import wraps
import torch
from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import Counter
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import Counter
import re

class QuestionStatistics(Toplevel):
    def __init__(self, parent, csv_file_path):
        super().__init__(parent)
        self.parent = parent
        self.title("Question Based Statistics")
        self.window_width = 1200
        self.window_height = 800
        self.geometry(f"{self.window_width}x{self.window_height}")
        self.resizable(True, True)
        
        # Get data from parent
        self.qa_data = pd.read_csv(csv_file_path)

        self.setup_ui()
        self.create_charts()
    
    def setup_ui(self):
        """Setup the UI with 4 quadrants"""
        # Main container
        main_frame = Frame(self, bg="white")
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Top row
        top_frame = Frame(main_frame, bg="white")
        top_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # Top left - Category pie chart
        self.top_left_frame = Frame(top_frame, bg="lightgray", relief=RAISED, bd=2)
        self.top_left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        Label(self.top_left_frame, text="Questions by Category", 
              font=("Arial", 12, "bold"), bg="lightgray").pack(pady=5)
        
        # Top right - Total count
        self.top_right_frame = Frame(top_frame, bg="lightblue", relief=RAISED, bd=2)
        self.top_right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        Label(self.top_right_frame, text="Total Q&A Statistics", 
              font=("Arial", 12, "bold"), bg="lightblue").pack(pady=5)
        
        # Bottom row
        bottom_frame = Frame(main_frame, bg="white")
        bottom_frame.pack(fill=BOTH, expand=True, pady=(5, 0))
        
        # Bottom left - Objects in answers (cars, pedestrians, etc.)
        self.bottom_left_frame = Frame(bottom_frame, bg="lightgreen", relief=RAISED, bd=2)
        self.bottom_left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        Label(self.bottom_left_frame, text="Objects in Answers", 
              font=("Arial", 12, "bold"), bg="lightgreen").pack(pady=5)
        
        # Bottom right - Question types (who, what, when, where, etc.)
        self.bottom_right_frame = Frame(bottom_frame, bg="lightyellow", relief=RAISED, bd=2)
        self.bottom_right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        Label(self.bottom_right_frame, text="Question Types", 
              font=("Arial", 12, "bold"), bg="lightyellow").pack(pady=5)
    
    def create_charts(self):
        """Create all 4 charts"""
        # Fix: Use .empty to check if DataFrame is empty
        if self.qa_data.empty:
            # Show "No data available" message in each quadrant
            for frame, bg_color in [(self.top_left_frame, "lightgray"), 
                                   (self.top_right_frame, "lightblue"),
                                   (self.bottom_left_frame, "lightgreen"), 
                                   (self.bottom_right_frame, "lightyellow")]:
                Label(frame, text="No data available", 
                      font=("Arial", 14), bg=bg_color).pack(expand=True)
            return
        
        self.create_category_pie_chart()
        self.create_total_count_display()
        self.create_objects_chart()
        self.create_question_types_chart()
    
    def create_category_pie_chart(self):
        """Top left - Pie chart of question categories"""
        # Count categories - Fix: Use .tolist() to convert to list
        categories = self.qa_data.get('category', 'Unknown').fillna('Unknown').tolist()
        category_counts = Counter(categories)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('lightgray')
        
        if category_counts:
            labels = list(category_counts.keys())
            sizes = list(category_counts.values())
            colors = plt.cm.Set3(range(len(labels)))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
            
            # Customize text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('Questions by Category', fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.top_left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    def create_total_count_display(self):
        """Top right - Total count statistics"""
        total_qa = len(self.qa_data)
        
        # Count unique videos - Fix: Use pandas methods
        video_columns = ["video_file_path", "video_file", "video_path", "file_name"]
        video_files = set()
        
        for col in video_columns:
            if col in self.qa_data.columns:
                unique_videos_in_col = self.qa_data[col].dropna().unique()
                video_files.update(unique_videos_in_col)
                break  # Use the first available column
        
        unique_videos = len(video_files)
        
        # Average Q&A per video
        avg_qa_per_video = total_qa / unique_videos if unique_videos > 0 else 0
        
        # Create display
        stats_frame = Frame(self.top_right_frame, bg="lightblue")
        stats_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Total Q&A
        Label(stats_frame, text=f"Total Q&A Pairs: {total_qa}", 
              font=("Arial", 16, "bold"), bg="lightblue", fg="darkblue").pack(pady=10)
        
        # Unique videos
        Label(stats_frame, text=f"Unique Videos: {unique_videos}", 
              font=("Arial", 14), bg="lightblue", fg="darkblue").pack(pady=5)
        
        # Average per video
        Label(stats_frame, text=f"Avg Q&A per Video: {avg_qa_per_video:.1f}", 
              font=("Arial", 14), bg="lightblue", fg="darkblue").pack(pady=5)
        
        # Categories count - Fix: Use pandas methods
        if 'category' in self.qa_data.columns:
            unique_categories = self.qa_data['category'].fillna('Unknown').nunique()
        else:
            unique_categories = 0
        
        Label(stats_frame, text=f"Question Categories: {unique_categories}", 
              font=("Arial", 14), bg="lightblue", fg="darkblue").pack(pady=5)
    
    def create_objects_chart(self):
        """Bottom left - Objects mentioned in answers (cars, pedestrians, etc.)"""
        # Keywords to search for in answers
        object_keywords = {
            'cars': ['car', 'cars', 'vehicle', 'vehicles', 'automobile', 'auto'],
            'pedestrians': ['pedestrian', 'pedestrians', 'people', 'person', 'walker', 'walkers'],
            'bicycles': ['bicycle', 'bicycles', 'bike', 'bikes', 'cyclist', 'cyclists'],
            'trucks': ['truck', 'trucks', 'lorry', 'lorries'],
            'buses': ['bus', 'buses'],
            'motorcycles': ['motorcycle', 'motorcycles', 'motorbike', 'motorbikes'],
            'traffic_lights': ['traffic light', 'traffic lights', 'signal', 'signals'],
            'buildings': ['building', 'buildings', 'structure', 'structures'],
            'roads': ['road', 'roads', 'street', 'streets', 'highway', 'highways']
        }
        
        # Count objects in answers - Fix: Use pandas methods
        object_counts = {}
        
        if 'answer' in self.qa_data.columns:
            answers = self.qa_data['answer'].fillna('').str.lower()
            
            for obj_type, keywords in object_keywords.items():
                # Count rows where any keyword appears in the answer
                count = 0
                for keyword in keywords:
                    count += answers.str.contains(keyword, regex=False).sum()
                
                if count > 0:
                    object_counts[obj_type.replace('_', ' ').title()] = count
        
        if object_counts:
            # Create bar chart
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('lightgreen')
            
            objects = list(object_counts.keys())
            counts = list(object_counts.values())
            colors = plt.cm.Set2(range(len(objects)))
            
            bars = ax.bar(objects, counts, color=colors)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Objects Mentioned in Answers', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.bottom_left_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)
        else:
            Label(self.bottom_left_frame, text="No objects detected in answers", 
                  font=("Arial", 12), bg="lightgreen").pack(expand=True)
    
    def create_question_types_chart(self):
        """Bottom right - Question types (who, what, when, where, etc.)"""
        # Question type patterns
        question_types = {
            'What': [r'^what\b', r'\bwhat\b'],
            'Who': [r'^who\b', r'\bwho\b'],
            'When': [r'^when\b', r'\bwhen\b'],
            'Where': [r'^where\b', r'\bwhere\b'],
            'Why': [r'^why\b', r'\bwhy\b'],
            'How': [r'^how\b', r'\bhow\b'],
            'Which': [r'^which\b', r'\bwhich\b'],
            'Can/Could': [r'^can\b', r'^could\b'],
            'Do/Does': [r'^do\b', r'^does\b', r'^did\b']
        }
        
        # Count question types - Fix: Use pandas methods
        type_counts = {}
        
        if 'question' in self.qa_data.columns:
            questions = self.qa_data['question'].fillna('').str.lower().str.strip()
            
            for q_type, patterns in question_types.items():
                count = 0
                for pattern in patterns:
                    count += questions.str.contains(pattern, regex=True, case=False).sum()
                
                if count > 0:
                    type_counts[q_type] = count
        
        if type_counts:
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('lightyellow')
            
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = plt.cm.Pastel1(range(len(types)))
            
            bars = ax.barh(types, counts, color=colors)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}', ha='left', va='center', fontweight='bold')
            
            ax.set_title('Question Types Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Count', fontweight='bold')
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.bottom_right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)
        else:
            Label(self.bottom_right_frame, text="No question patterns detected", 
                  font=("Arial", 12), bg="lightyellow").pack(expand=True)

class CircleSelector(Toplevel):
    def __init__(self, parent, pil_image, on_confirm):
        super().__init__(parent)
        self.title("Draw Circle to Blur Object")
        self.on_confirm = on_confirm
        
        # Fixed window dimensions
        self.window_width = 900
        self.window_height = 800
        self.canvas_width = 780
        self.canvas_height = 500
        
        # Set window size and center it
        self.geometry(f"{self.window_width}x{self.window_height}")
        self.resizable(False, False)
        
        # Store the original image and create scaled version
        self.original_image = pil_image
        self.scale_factor_x = pil_image.width / self.canvas_width
        self.scale_factor_y = pil_image.height / self.canvas_height
        
        # Scale image to fit canvas while maintaining aspect ratio
        self.scaled_image = self.scale_image_to_fit(pil_image)
        
        # Instructions at top
        instruction_frame = Frame(self, bg="lightblue", height=80)
        instruction_frame.pack(fill=X, pady=5, padx=10)
        instruction_frame.pack_propagate(False)
        
        Label(instruction_frame, text="Instructions:", font=("Arial", 11, "bold"), 
              bg="lightblue").pack()
        Label(instruction_frame, text="1. Click and drag to draw a circle around the object", 
              font=("Arial", 10), bg="lightblue").pack()
        Label(instruction_frame, text="2. Release mouse button to finish drawing", 
              font=("Arial", 10), bg="lightblue").pack()
        Label(instruction_frame, text="3. You can draw multiple circles to select different objects", 
              font=("Arial", 10), bg="lightblue").pack()
        
        # Create canvas with fixed size
        canvas_frame = Frame(self, relief=SUNKEN, bd=2)
        canvas_frame.pack(pady=10, padx=10)
        
        self.canvas = Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                           bg="white", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack()
        
        # Display the scaled image
        self.img_tk = ImageTk.PhotoImage(self.scaled_image)
        self.canvas.create_image(self.canvas_width//2, self.canvas_height//2, image=self.img_tk)
        
        # Circle drawing state
        self.circles = []  # Store all circles as (center, radius, canvas_object)
        self.current_circle = None
        self.is_drawing = False
        self.start_pos = None
        
        # Status label
        self.status_var = StringVar(value="Ready - Click and drag to draw circle(s)")
        self.status_label = Label(self, textvariable=self.status_var, font=("Arial", 10), 
                                fg="blue", bg="lightyellow", relief=SUNKEN)
        self.status_label.pack(fill=X, pady=5, padx=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_circle)
        self.canvas.bind("<B1-Motion>", self.draw_circle)
        self.canvas.bind("<ButtonRelease-1>", self.end_circle)
        
        # Control buttons
        button_frame = Frame(self)
        button_frame.pack(pady=15)
        
        self.process_btn = Button(button_frame, text="Process Video", command=self.confirm,
                                bg="lightgreen", font=("Arial", 11, "bold"), width=15, height=2)
        self.process_btn.pack(side=LEFT, padx=10)
        self.process_btn.config(state=DISABLED)  # Disabled until circle is drawn
        
        self.clear_btn = Button(button_frame, text="Clear All Circles", command=self.clear_all_circles,
                              bg="lightyellow", font=("Arial", 11), width=15, height=2)
        self.clear_btn.pack(side=LEFT, padx=10)
        self.clear_btn.config(state=DISABLED)
        
        self.undo_btn = Button(button_frame, text="Undo Last", command=self.undo_last_circle,
                             bg="lightcyan", font=("Arial", 11), width=15, height=2)
        self.undo_btn.pack(side=LEFT, padx=10)
        self.undo_btn.config(state=DISABLED)
        
        self.cancel_btn = Button(button_frame, text="Cancel", command=self.destroy,
                               bg="lightcoral", font=("Arial", 11), width=15, height=2)
        self.cancel_btn.pack(side=LEFT, padx=10)
        
        # Center the window
        self.center_window()

    def scale_image_to_fit(self, image):
        """Scale image to fit canvas while maintaining aspect ratio"""
        img_width, img_height = image.size
        
        # Calculate scale factor to fit within canvas
        scale_x = self.canvas_width / img_width
        scale_y = self.canvas_height / img_height
        scale = min(scale_x, scale_y)  # Use smaller scale to maintain aspect ratio
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Update scale factors for coordinate conversion
        self.actual_scale_x = img_width / new_width
        self.actual_scale_y = img_height / new_height
        
        return scaled_image

    def center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.window_width // 2)
        y = (self.winfo_screenheight() // 2) - (self.window_height // 2)
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def start_circle(self, event):
        """Start drawing a new circle"""
        self.start_pos = (event.x, event.y)
        self.is_drawing = True
        
        # Create new circle
        self.current_circle = self.canvas.create_oval(
            event.x, event.y, event.x, event.y, 
            outline="red", width=3, fill="", stipple=""
        )
        
        self.status_var.set("Drawing circle... Release to finish")

    def draw_circle(self, event):
        """Update circle while dragging"""
        if not self.is_drawing or not self.start_pos:
            return
            
        # Calculate radius
        dx = event.x - self.start_pos[0]
        dy = event.y - self.start_pos[1]
        radius = (dx**2 + dy**2)**0.5
        
        # Update circle coordinates
        x0 = self.start_pos[0] - radius
        y0 = self.start_pos[1] - radius
        x1 = self.start_pos[0] + radius
        y1 = self.start_pos[1] + radius
        
        self.canvas.coords(self.current_circle, x0, y0, x1, y1)

    def end_circle(self, event):
        """Finish drawing current circle"""
        if not self.is_drawing or not self.start_pos:
            return
            
        # Calculate final radius
        dx = event.x - self.start_pos[0]
        dy = event.y - self.start_pos[1]
        radius = (dx**2 + dy**2)**0.5
        
        # Only keep circles with minimum radius
        if radius >= 5:
            # Convert canvas coordinates to original image coordinates
            original_center = (
                int(self.start_pos[0] * self.actual_scale_x),
                int(self.start_pos[1] * self.actual_scale_y)
            )
            original_radius = int(radius * max(self.actual_scale_x, self.actual_scale_y))
            
            # Store circle info: (center, radius, canvas_object)
            self.circles.append((original_center, original_radius, self.current_circle))
            
            # Update status
            self.status_var.set(f"Circle {len(self.circles)} added - Click and drag to add more circles")
            
            # Enable buttons
            self.update_button_states()
        else:
            # Remove circle if too small
            self.canvas.delete(self.current_circle)
            self.status_var.set("Circle too small - Try again with a larger circle")
        
        # Reset drawing state
        self.is_drawing = False
        self.start_pos = None
        self.current_circle = None

    def clear_all_circles(self):
        """Clear all circles"""
        for _, _, canvas_obj in self.circles:
            self.canvas.delete(canvas_obj)
        
        self.circles.clear()
        self.status_var.set("All circles cleared - Click and drag to draw new circles")
        self.update_button_states()

    def undo_last_circle(self):
        """Remove the last drawn circle"""
        if self.circles:
            _, _, canvas_obj = self.circles.pop()
            self.canvas.delete(canvas_obj)
            
            if self.circles:
                self.status_var.set(f"{len(self.circles)} circle(s) remaining - Click and drag to add more")
            else:
                self.status_var.set("All circles removed - Click and drag to draw new circles")
            
            self.update_button_states()

    def update_button_states(self):
        """Update button states based on current circles"""
        has_circles = len(self.circles) > 0
        
        # Enable/disable buttons based on whether we have circles
        self.process_btn.config(state=NORMAL if has_circles else DISABLED)
        self.clear_btn.config(state=NORMAL if has_circles else DISABLED)
        self.undo_btn.config(state=NORMAL if has_circles else DISABLED)
        
        # Update process button text to show number of circles
        if has_circles:
            self.process_btn.config(text=f"✓ Process {len(self.circles)} Circle(s)")
        else:
            self.process_btn.config(text="✓ Process Video")

    def confirm(self):
        """Confirm selection and start processing"""
        if not self.circles:
            messagebox.showerror("Error", "Please draw at least one circle first")
            return
            
        # Show confirmation dialog
        circle_count = len(self.circles)
        result = messagebox.askyesno(
            "Confirm Processing", 
            f"Ready to process video with {circle_count} selected region(s)?\n\n"
            "This will blur all selected areas using SAM segmentation.\n"
            "Processing may take several minutes depending on video length."
        )
        
        if result:
            # Pass all circles to the callback
            self.on_confirm(self.circles)
            self.destroy()
        else:
            self.status_var.set(f"{circle_count} circle(s) ready - Click 'Process' when ready")

def run_sam_on_circle(image_bgr, center, radius, model_path=SAM_CHECKPOINT_PATH, device='cpu'):
    """Run SAM model on the selected circle area"""
    try:
        # Load SAM model
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device)
        predictor = SamPredictor(sam)

        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Create input point and box from circle
        input_point = [center]
        input_label = [1]  # 1 means "include this object"

        # Create bounding box from circle
        box = [
            max(0, center[0] - radius),
            max(0, center[1] - radius),
            min(image_bgr.shape[1], center[0] + radius),
            min(image_bgr.shape[0], center[1] + radius)
        ]

        # Get mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            box=np.array(box)[None, :],
            multimask_output=False,
        )
        
        # Return the best mask
        return masks[0] if len(masks) > 0 else None
        
    except Exception as e:
        print(f"Error running SAM: {e}")
        return None


class AsyncVideoAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotation Tool ")
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
        
        Button(self.right_panel, text="Split Videos", bg="lightblue", font=("Arial", 9, "bold"), 
               width=18, height=2, command=self.split_videos).pack(pady=10)
        Button(self.right_panel, text="Load Existing CSV", bg="lightyellow", font=("Arial", 9, "bold"), 
               width=18, height=2, command=self.load_csv_async).pack(pady=5)
        
        Label(self.right_panel, text="Current CSV:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.csv_status_var = StringVar(value="None loaded")
        Label(self.right_panel, textvariable=self.csv_status_var, bg="lightgray", 
              font=("Arial", 8), wraplength=180).pack(pady=5, padx=10)

        
        Label(self.right_panel, text="Export Options:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        Button(self.right_panel, text="Blur and Track", command=self.blur_track, width=18).pack(pady=5)
        Button(self.right_panel, text="Question Statistics", command=self.statistics, width=18).pack(pady=5)
        
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
        self.frame_label.update_idletasks()
        display_width = self.frame_label.winfo_width()
        display_height = self.frame_label.winfo_height()
        if display_width <= 1 or display_height <= 1:
            display_width = 800  # Adjust based on your center panel width
            display_height = 600  # Adjust based on your center panel height
        img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
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
            
            # Get the actual size of the frame_label widget
            self.frame_label.update_idletasks()
            display_width = self.frame_label.winfo_width()
            display_height = self.frame_label.winfo_height()
            
            # If widget hasn't been rendered yet, use default values
            if display_width <= 1 or display_height <= 1:
                display_width = 800  # Adjust based on your center panel width
                display_height = 600  # Adjust based on your center panel height
            
            img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
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
                
                # Get the actual size of the frame_label widget
                self.frame_label.update_idletasks()
                display_width = self.frame_label.winfo_width()
                display_height = self.frame_label.winfo_height()
                
                # If widget hasn't been rendered yet, use default values
                if display_width <= 1 or display_height <= 1:
                    display_width = 800  # Adjust based on your center panel width
                    display_height = 600  # Adjust based on your center panel height
                
                img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
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

    def split_videos(self):
        # Ask for source folder (videos)
        source_directory = filedialog.askdirectory(title="Select folder containing videos")
        if not source_directory:
            return
        self.base_directory = source_directory
        self.video_dir = source_directory
        self.dir_var.set(source_directory)
        self.populate_video_list()

        # Ask for output folder (where clips will be saved)
        output_base_directory = filedialog.askdirectory(title="Select folder to save split video clips")
        if not output_base_directory:
            return

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv','.moov')
        videos = []

        try:
            for file in os.listdir(self.base_directory):
                if file.lower().endswith(video_extensions):
                    videos.append(file)

            for video in videos:
                video_path = os.path.join(self.base_directory, video)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    messagebox.showerror("Error", f"Could not open video file: {video}")
                    continue

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_per_clip = 10 * fps  # 10-second clips

                # Output: Create a subfolder for each video inside the output base directory
                output_dir = os.path.join(output_base_directory, f"{os.path.splitext(video)[0]}_clips")
                os.makedirs(output_dir, exist_ok=True)

                clip_num = 0
                start_frame = 0

                while start_frame < total_frames:
                    end_frame = min(start_frame + frames_per_clip, total_frames)
                    output_filename = os.path.join(output_dir, f"clip_{clip_num:03d}.mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    for i in range(start_frame, end_frame):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)

                    print(f"Clip {clip_num} saved: {output_filename}")
                    out.release()
                    start_frame = end_frame
                    clip_num += 1

                cap.release()
                cv2.destroyAllWindows()

            self.status_var.set(f"Processed {len(videos)} videos.")
            print(f"Processed {len(videos)} video files.")

        except Exception as e:
            messagebox.showerror("Error", f"Could not process videos: {str(e)}")

    def blur_track(self):
        """Start the blur and track process"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video first.")
            return
            
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            messagebox.showerror("SAM Model Missing", 
                               f"SAM checkpoint not found at: {SAM_CHECKPOINT_PATH}\n"
                               "Please download the SAM model checkpoint.")
            return
        
        try:
            # Get first frame for circle selection
            cap = cv2.VideoCapture(self.current_video_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                messagebox.showerror("Error", "Could not read first frame from video.")
                return
            
            # Convert first frame to PIL Image for circle selector
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(first_frame_rgb)
            
            # Show circle selector
            def on_circles_confirm(circles):
                print(f"Multiple circles selected: {len(circles)} circles")
                for i, (center, radius, _) in enumerate(circles):
                    print(f"Circle {i+1} - Center: {center}, Radius: {radius}")
                self.start_blur_and_track_process_multiple(circles)
            
            CircleSelector(self.root, pil_image, on_circles_confirm)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start blur and track: {str(e)}")

    def start_blur_and_track_process_multiple(self, circles):
        """Start the blur and track process for multiple circles in a separate thread"""
        self.status_var.set("Starting SAM blur and track for multiple objects...")
        
        def blur_track_thread():
            try:
                self.blur_and_track_yolo_sam_predictive(circles)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Blur and track failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Blur and track failed"))
        
        threading.Thread(target=blur_track_thread, daemon=True).start()
    def blur_and_track_yolo_sam_predictive(self, circles):

        circle_count = len(circles)
        print(f"Starting predictive YOLO + SAM blur and track for {circle_count} circles")
        
        # Load YOLO model
        try:
            yolo_model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {e}. Please install ultralytics: pip install ultralytics")
        
        # Load video
        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties - FPS: {fps}, Size: {frame_width}x{frame_height}, Frames: {total_frames}")
        
        # Prepare output video
        output_path = os.path.splitext(self.current_video_path)[0] + f"_predictive_blur_{circle_count}objects.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            cap.release()
            raise Exception("Could not create output video file")
        
        # Load SAM model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        try:
            sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(device)
            predictor = SamPredictor(sam)
            print("SAM model loaded successfully")
        except Exception as e:
            cap.release()
            out.release()
            raise Exception(f"Failed to load SAM model: {e}")
        
        # Update status
        self.root.after(0, lambda: self.status_var.set(f"Initializing predictive tracking for {circle_count} objects..."))
        
        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            raise Exception("Could not read first frame")
        
        # Initialize object tracking data with velocity tracking
        objects_data = []
        
        # Run YOLO on first frame
        results = yolo_model(first_frame, verbose=False)
        detections = results[0].boxes
        
        if detections is not None:
            boxes = detections.xyxy.cpu().numpy()
            confidences = detections.conf.cpu().numpy()
            class_ids = detections.cls.cpu().numpy().astype(int)
            
            print(f"YOLO detected {len(boxes)} objects in first frame")
            
            # Match circles with YOLO detections
            for i, (center, radius, _) in enumerate(circles):
                best_match_idx = None
                best_distance = float('inf')
                
                for j, (x1, y1, x2, y2) in enumerate(boxes):
                    box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    distance = ((center[0] - box_center[0])**2 + (center[1] - box_center[1])**2)**0.5
                    
                    if distance < best_distance and distance < radius * 2:
                        best_distance = distance
                        best_match_idx = j
                
                if best_match_idx is not None:
                    x1, y1, x2, y2 = boxes[best_match_idx]
                    class_id = class_ids[best_match_idx]
                    confidence = confidences[best_match_idx]
                    class_name = yolo_model.names[class_id] if class_id < len(yolo_model.names) else f"class_{class_id}"
                    
                    print(f"Circle {i+1} matched with {class_name} (conf: {confidence:.3f})")
                    
                    # Get initial SAM mask
                    box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    sam_mask = self._run_sam_on_box(first_frame, [x1, y1, x2, y2], predictor)
                    
                    if sam_mask is not None:
                        mask = (sam_mask.astype(np.uint8) * 255)
                        
                        obj_data = {
                            'original_center': center,
                            'original_radius': radius,
                            'current_center': box_center,
                            'predicted_center': box_center,  # For prediction
                            'current_bbox': [x1, y1, x2, y2],
                            'predicted_bbox': [x1, y1, x2, y2],  # For prediction
                            'current_mask': mask,
                            'predicted_mask': mask.copy(),  # Mask at predicted position
                            'class_id': class_id,
                            'class_name': class_name,
                            'track_id': i,
                            'lost_tracking': False,
                            'frames_lost': 0,
                            # Velocity tracking
                            'velocity': (0, 0),
                            'position_history': [box_center],  # Last 3 positions for velocity calc
                            'acceleration': (0, 0),
                            'mask_update_counter': 0
                        }
                        objects_data.append(obj_data)
                        print(f"Initialized predictive tracking for {class_name} at {box_center}")
        
        if not objects_data:
            print("No YOLO matches found, falling back to SAM-only approach")
            for i, (center, radius, _) in enumerate(circles):
                mask = run_sam_on_circle(first_frame, center, radius, model_path=SAM_CHECKPOINT_PATH, device=device)
                if mask is not None:
                    mask = mask.astype(np.uint8) * 255
                    obj_data = {
                        'original_center': center,
                        'original_radius': radius,
                        'current_center': center,
                        'predicted_center': center,
                        'current_bbox': None,
                        'predicted_bbox': None,
                        'current_mask': mask,
                        'predicted_mask': mask.copy(),
                        'class_id': -1,
                        'class_name': 'unknown',
                        'track_id': i,
                        'lost_tracking': False,
                        'frames_lost': 0,
                        'velocity': (0, 0),
                        'position_history': [center],
                        'acceleration': (0, 0),
                        'mask_update_counter': 0
                    }
                    objects_data.append(obj_data)
        
        print(f"Tracking {len(objects_data)} objects with predictive positioning")
        
        # Process first frame
        combined_mask = self._combine_masks([obj['predicted_mask'] for obj in objects_data])
        blurred_frame = self._blur_mask(first_frame, combined_mask)
        out.write(blurred_frame)
        
        frame_count = 1
        
        self.root.after(0, lambda: self.status_var.set(f"Processing with predictive blur for {len(objects_data)} objects..."))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            if frame_count % 30 == 0:
                active_count = len([obj for obj in objects_data if not obj['lost_tracking']])
                progress = f"Frame {frame_count}/{total_frames} - Predictive tracking {active_count}/{len(objects_data)} objects"
                self.root.after(0, lambda p=progress: self.status_var.set(p))
                print(progress)
            
            # STEP 1: Predict where objects should be based on velocity
            for obj_data in objects_data:
                if obj_data['lost_tracking']:
                    continue
                
                # Calculate predicted position using velocity
                current_pos = obj_data['current_center']
                velocity = obj_data['velocity']
                
                # Predict next position
                predicted_x = int(current_pos[0] + velocity[0])
                predicted_y = int(current_pos[1] + velocity[1])
                
                # Keep within frame bounds
                predicted_x = max(50, min(frame_width - 50, predicted_x))
                predicted_y = max(50, min(frame_height - 50, predicted_y))
                
                obj_data['predicted_center'] = (predicted_x, predicted_y)
                
                # Update predicted bounding box if we have one
                if obj_data['current_bbox'] is not None:
                    x1, y1, x2, y2 = obj_data['current_bbox']
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Move bbox to predicted position
                    new_x1 = predicted_x - bbox_width // 2
                    new_y1 = predicted_y - bbox_height // 2
                    new_x2 = new_x1 + bbox_width
                    new_y2 = new_y1 + bbox_height
                    
                    obj_data['predicted_bbox'] = [new_x1, new_y1, new_x2, new_y2]
                
                # Move the mask to predicted position
                obj_data['predicted_mask'] = self._move_mask_to_position(
                    obj_data['current_mask'], 
                    obj_data['current_center'], 
                    obj_data['predicted_center']
                )
            
            # STEP 2: Run YOLO detection (this happens after prediction, so no lag)
            results = yolo_model(frame, verbose=False)
            current_detections = results[0].boxes
            
            if current_detections is not None:
                current_boxes = current_detections.xyxy.cpu().numpy()
                current_confidences = current_detections.conf.cpu().numpy()
                current_class_ids = current_detections.cls.cpu().numpy().astype(int)
                
                # STEP 3: Update actual positions and calculate velocity
                for obj_data in objects_data:
                    if obj_data['lost_tracking']:
                        continue
                    
                    # Find best matching detection
                    best_match_idx = None
                    best_score = 0
                    
                    for j, (x1, y1, x2, y2) in enumerate(current_boxes):
                        if obj_data['class_id'] >= 0 and current_class_ids[j] != obj_data['class_id']:
                            continue
                        
                        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        prev_center = obj_data['current_center']
                        
                        distance = ((box_center[0] - prev_center[0])**2 + (box_center[1] - prev_center[1])**2)**0.5
                        max_distance = 150  # Allow larger movements
                        
                        if distance < max_distance:
                            score = (max_distance - distance) / max_distance * current_confidences[j]
                            if score > best_score:
                                best_score = score
                                best_match_idx = j
                    
                    if best_match_idx is not None and best_score > 0.2:
                        # Update actual position
                        x1, y1, x2, y2 = current_boxes[best_match_idx]
                        new_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        old_center = obj_data['current_center']
                        
                        # Calculate actual movement and velocity
                        movement = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                        movement_distance = (movement[0]**2 + movement[1]**2)**0.5
                        
                        # Update velocity (smoothed)
                        old_velocity = obj_data['velocity']
                        new_velocity = (
                            0.7 * old_velocity[0] + 0.3 * movement[0],  # Smooth velocity
                            0.7 * old_velocity[1] + 0.3 * movement[1]
                        )
                        
                        obj_data['current_center'] = new_center
                        obj_data['current_bbox'] = [x1, y1, x2, y2]
                        obj_data['velocity'] = new_velocity
                        obj_data['frames_lost'] = 0
                        obj_data['mask_update_counter'] += 1
                        
                        # Add to position history
                        obj_data['position_history'].append(new_center)
                        if len(obj_data['position_history']) > 3:
                            obj_data['position_history'].pop(0)
                        
                        print(f"Frame {frame_count}: {obj_data['class_name']} - Actual: {movement}, Velocity: ({new_velocity[0]:.1f}, {new_velocity[1]:.1f})")
                        
                        # Update SAM mask every few frames or on large movements
                        if obj_data['mask_update_counter'] % 3 == 0 or movement_distance > 25:
                            sam_mask = self._run_sam_on_box(frame, [x1, y1, x2, y2], predictor)
                            if sam_mask is not None:
                                obj_data['current_mask'] = (sam_mask.astype(np.uint8) * 255)
                                # Also update predicted mask to current position
                                obj_data['predicted_mask'] = obj_data['current_mask'].copy()
                                print(f"Updated SAM mask for {obj_data['class_name']}")
                    
                    else:
                        # Object not found, but keep using predicted position
                        obj_data['frames_lost'] += 1
                        print(f"Frame {frame_count}: Using prediction for {obj_data['class_name']} ({obj_data['frames_lost']} frames)")
                        
                        if obj_data['frames_lost'] > 15:  # Allow more frames before giving up
                            obj_data['lost_tracking'] = True
                            print(f"Marking {obj_data['class_name']} as permanently lost")
            
            # STEP 4: Apply blur using PREDICTED positions (no lag!)
            active_masks = [obj['predicted_mask'] for obj in objects_data if not obj['lost_tracking']]
            
            if active_masks:
                combined_mask = self._combine_masks(active_masks)
                blurred_frame = self._blur_mask(frame, combined_mask)
            else:
                blurred_frame = frame
            
            out.write(blurred_frame)
        
        # Cleanup
        cap.release()
        out.release()
    
    # Update UI
        def completion_message():
            active_objects = sum(1 for obj in objects_data if not obj['lost_tracking'])
            self.status_var.set("Predictive blur tracking completed!")
            messagebox.showinfo("Complete", 
                            f"Predictive YOLO + SAM tracking completed!\n"
                            f"Processed {frame_count} frames\n"
                            f"Final active objects: {active_objects}/{len(objects_data)}\n\n"
                            f"Objects tracked:\n" + 
                            "\n".join([f"- {obj['class_name']}" for obj in objects_data]) +
                            f"\n\nOutput saved as:\n{output_path}")
        
        self.root.after(0, completion_message)
        print(f"Predictive blur tracking completed. Output saved to: {output_path}")

    def _move_mask_to_position(self, mask, old_center, new_center):
        """Move a mask from old position to new position"""
        if old_center == new_center:
            return mask.copy()
        
        # Calculate translation
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Apply translation to mask
        moved_mask = cv2.warpAffine(
            mask, 
            translation_matrix, 
            (mask.shape[1], mask.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return moved_mask

    def _run_sam_on_box(self, frame, bbox, predictor):
        """Run SAM segmentation on a bounding box"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(frame_rgb)
            
            input_box = np.array(bbox)
            
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            return masks[0] if len(masks) > 0 else None
            
        except Exception as e:
            print(f"Error running SAM on box: {e}")
            return None
    

    def _combine_masks(self, masks):
                """Combine multiple masks into a single mask"""
                if not masks:
                    return None
                    
                if len(masks) == 1:
                    return masks[0]
                
                # Start with first mask
                combined = masks[0].copy()
                
                # Add all other masks using logical OR
                for mask in masks[1:]:
                    combined = cv2.bitwise_or(combined, mask)
                
                return combined
    def _blur_mask(self, frame, mask):
            """Apply blur to masked region of frame"""
            # Create a strong blur
            blurred = cv2.GaussianBlur(frame, (41, 41), 0)
            
            # Create 3-channel mask
            mask_3ch = cv2.merge([mask, mask, mask])
            
            # Apply mask: where mask is 255, use blurred; otherwise use original
            result = np.where(mask_3ch == 255, blurred, frame)
            
            return result.astype(np.uint8)

    def statistics(self):
            """Statistic function that shows all statistics based on various question answers"""
            print("csv file path",self.csv_file_path)
            if self.csv_file_path == "":
                messagebox.showinfo("Statistics","Questions not found")
                return
            QuestionStatistics(self.root,self.csv_file_path)

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