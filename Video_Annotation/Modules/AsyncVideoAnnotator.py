import cv2
import os
import shutil
import datetime

import ttkbootstrap as ttk
from ttkbootstrap import Style
import re
import shutil
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
from Modules.QuestionStats import QuestionStatistics
from Modules.CircleSelect import run_sam_on_circle, CircleSelector



SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"



class AsyncVideoAnnotationTool:
    def __init__(self, root):
        self.style = Style(theme="pulse")
        self.root = root
        self.root.title("Video Annotation Tool ")
        self.root.geometry("1200x800")

        # State
        self.auto_play_enabled = True
        self.questions_disabled = True
        self.qa_states = {}
        self.qa_history = []
        self.editable_questions = set()
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
    def get_qa_selection(self):
        """Compatibility wrapper for getting selections"""
        if hasattr(self.qa_listbox, 'selection'):  # Treeview
            return self.qa_listbox.selection()
        else:  # Original Listbox
            return self.qa_listbox.curselection()

    def delete_qa_items(self, start, end=None):
        """Compatibility wrapper for deleting items"""
        if hasattr(self.qa_listbox, 'get_children'):  # Treeview
            children = self.qa_listbox.get_children()
            if children:  # Only delete if there are items
                self.qa_listbox.delete(*children)
        else:  # Original Listbox
            if end is None:
                self.qa_listbox.delete(start)
            else:
                self.qa_listbox.delete(start, end)

    def insert_qa_item(self, position, text=None, values=None):
        """Compatibility wrapper for inserting items"""
        if hasattr(self.qa_listbox, 'heading'):  # Treeview (better check)
            if values:
                return self.qa_listbox.insert('', position, values=values)
            else:
                # Convert text to values format for Treeview
                return self.qa_listbox.insert('', position, values=(text or "", "", "", "None"))
        else:  # Original Listbox
            return self.qa_listbox.insert(position, text or "")

    def disable_question_interaction(self):
        """Disable question selection and buttons"""
        if hasattr(self.qa_listbox, 'state'):  # Treeview
            self.qa_listbox.state(['disabled'])
        else:  # Original Listbox
            self.qa_listbox.config(state='disabled')
        
        self.accept_btn.config(state=DISABLED)
        self.reject_btn.config(state=DISABLED)
        self.reset_btn.config(state=DISABLED)
        self.questions_disabled = True

    def enable_question_interaction(self):
        """Enable question selection and buttons"""
        if hasattr(self.qa_listbox, 'state'):  # Treeview
            self.qa_listbox.state(['!disabled'])
        else:  # Original Listbox
            self.qa_listbox.config(state='normal')
        
        self.questions_disabled = False
        self.update_button_states()

    def on_qa_selection_change(self, event=None):
        """Handle selection changes in QA list"""
        self.update_button_states()

    def update_button_states(self):
        """Update button states based on current selection"""
        if self.questions_disabled:
            return
            
        selection = self.get_qa_selection()
        if selection:
            if hasattr(self.qa_listbox, 'selection'):  # Treeview
                item_id = selection[0] if selection else None
                current_state = self.qa_states.get(item_id, 'none') if item_id else 'none'
            else:  # Original Listbox
                current_state = 'none'  # Fallback for original listbox
            
            if current_state == 'none':
                self.accept_btn.config(state=NORMAL)
                self.reject_btn.config(state=NORMAL)
                self.reset_btn.config(state=DISABLED)
            else:
                self.accept_btn.config(state=DISABLED)
                self.reject_btn.config(state=DISABLED)
                self.reset_btn.config(state=NORMAL)
        else:
            self.accept_btn.config(state=DISABLED)
            self.reject_btn.config(state=DISABLED)
            self.reset_btn.config(state=DISABLED)

    def show_qa_for_current_video(self):
        """Enhanced version to populate with saved status from CSV"""
        # Clear existing items using compatibility wrapper
        self.delete_qa_items(0, END)
        
        self.current_video_qa = []
        
        if not self.current_video_path or not self.qa_data:
            self.accepted_var.set("0")
            self.rejected_var.set("0")
            return
            
        video_file_name = os.path.basename(self.current_video_path)
        
        for i, qa in enumerate(self.qa_data):
            video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                        qa.get("video_path") or qa.get("file_name") or "")
            if video_file == video_file_name:
                self.current_video_qa.append(qa)
                category = qa.get("category", "Unknown")
                question = qa.get("question", "No question")
                answer = qa.get("answer", "No answer")
                status = qa.get("status", "none")
                
                if hasattr(self.qa_listbox, 'insert') and hasattr(self.qa_listbox, 'heading'):  # Treeview
                    # Insert into treeview with saved status
                    status_display = status.capitalize() if status != 'none' else 'None'
                    item_id = self.insert_qa_item('end', values=(category, question, answer, status_display))
                    
                    # Set the state and color based on saved status
                    self.qa_states[item_id] = status
                    if status == 'accepted':
                        self.qa_listbox.item(item_id, tags=('accepted',))
                        self.qa_listbox.tag_configure('accepted', background='lightgreen')
                    elif status == 'rejected':
                        self.qa_listbox.item(item_id, tags=('rejected',))
                        self.qa_listbox.tag_configure('rejected', background='lightcoral')
                        
                else:  # Original Listbox mode
                    qtext = f'[{category}] {question} (Ans: {answer})'
                    self.insert_qa_item(END, text=qtext)
        
        # Update counters based on saved status
        self.update_counters_from_data()
        
        print(f"Loaded {len(self.current_video_qa)} Q&A items for {video_file_name}")

    def edit_qa_item(self, event):
        """FIXED: Handle double-click editing of QA items"""
        if self.questions_disabled:
            return
        
        selection = self.qa_listbox.selection()
        if not selection:
            return
            
        item = selection[0]
        
        # Only allow editing if item is rejected
        if self.qa_states.get(item, 'none') != 'rejected':
            messagebox.showinfo("Info", "Only rejected items can be edited.\n\n"
                            "Please reject an item first, then double-click to edit it.")
            return
        
        # Get item values - FIXED: Use qa_listbox instead of qa_tree
        values = self.qa_listbox.item(item, 'values')
        
        # Create edit dialog
        self.create_edit_dialog(item, values)
    def create_edit_dialog(self, item_id, values):
        
        edit_window = Toplevel(self.root)
        edit_window.title("Edit Q&A Item")
        edit_window.geometry("1200x960")
        edit_window.resizable(True, True)
        
        # Make dialog modal
        edit_window.transient(self.root)
       
        
        # Center the dialog
        edit_window.update_idletasks()
        x = (edit_window.winfo_screenwidth() // 2) - (300)
        y = (edit_window.winfo_screenheight() // 2) - (200)
        edit_window.geometry(f"600x400+{x}+{y}")
        edit_window.grab_set()
        
        # Main frame with padding
        main_frame = Frame(edit_window)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Category section
        Label(main_frame, text="Category:", font=("Arial", 10, "bold")).pack(anchor=W, pady=(0, 5))
        category_var = StringVar(value=values[0] if len(values) > 0 else "")
        category_entry = Entry(main_frame, textvariable=category_var, width=70, font=("Arial", 10))
        category_entry.pack(fill=X, pady=(0, 10))
        
        # Question section
        Label(main_frame, text="Question:", font=("Arial", 10, "bold")).pack(anchor=W, pady=(0, 5))
        question_frame = Frame(main_frame)
        question_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        question_text = Text(question_frame, height=6, width=70, font=("Arial", 10), wrap=WORD)
        question_scrollbar = Scrollbar(question_frame, orient=VERTICAL, command=question_text.yview)
        question_text.config(yscrollcommand=question_scrollbar.set)
        
        question_text.pack(side=LEFT, fill=BOTH, expand=True)
        question_scrollbar.pack(side=RIGHT, fill=Y)
        
        if len(values) > 1:
            question_text.insert(1.0, values[1])
        
        # Answer section
        Label(main_frame, text="Answer:", font=("Arial", 10, "bold")).pack(anchor=W, pady=(0, 5))
        answer_frame = Frame(main_frame)
        answer_frame.pack(fill=BOTH, expand=True, pady=(0, 15))
        
        answer_text = Text(answer_frame, height=6, width=70, font=("Arial", 10), wrap=WORD)
        answer_scrollbar = Scrollbar(answer_frame, orient=VERTICAL, command=answer_text.yview)
        answer_text.config(yscrollcommand=answer_scrollbar.set)
        
        answer_text.pack(side=LEFT, fill=BOTH, expand=True)
        answer_scrollbar.pack(side=RIGHT, fill=Y)
        
        if len(values) > 2:
            answer_text.insert(1.0, values[2])
        
        # Buttons frame
        btn_frame = Frame(main_frame)
        btn_frame.pack(fill=X, pady=(10, 0))
        
        def save_changes():
            try:
                    new_category = category_var.get().strip()
                    new_question = question_text.get(1.0, END).strip()
                    new_answer = answer_text.get(1.0, END).strip()
                    
                    # Validate input
                    if not new_category:
                        messagebox.showwarning("Validation Error", "Category cannot be empty.")
                        return
                    if not new_question:
                        messagebox.showwarning("Validation Error", "Question cannot be empty.")
                        return
                    if not new_answer:
                        messagebox.showwarning("Validation Error", "Answer cannot be empty.")
                        return
                    
                    # Update treeview - automatically set to accepted after edit
                    self.qa_listbox.item(item_id, values=(new_category, new_question, new_answer, 'Accepted'))
                    
                    # Update UI state
                    self.qa_states[item_id] = 'accepted'
                    self.qa_listbox.item(item_id, tags=('accepted',))
                    self.qa_listbox.tag_configure('accepted', background='lightgreen')
                    
                    # Update underlying data in current_video_qa
                    try:
                        all_items = self.qa_listbox.get_children()
                        item_index = list(all_items).index(item_id)
                        
                        if 0 <= item_index < len(self.current_video_qa):
                            # Update the data
                            self.current_video_qa[item_index]['category'] = new_category
                            self.current_video_qa[item_index]['question'] = new_question
                            self.current_video_qa[item_index]['answer'] = new_answer
                            self.current_video_qa[item_index]['status'] = 'accepted'  # Auto-accept after edit
                            
                            # Also update in main qa_data
                            video_file_name = os.path.basename(self.current_video_path) if self.current_video_path else ""
                            for qa in self.qa_data:
                                video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                                            qa.get("video_path") or qa.get("file_name") or "")
                                if (video_file == video_file_name and 
                                    qa.get('question') == values[1] and  # Match original question
                                    qa.get('answer') == values[2]):     # Match original answer
                                    qa['category'] = new_category
                                    qa['question'] = new_question
                                    qa['answer'] = new_answer
                                    qa['status'] = 'accepted'  # Auto-accept after edit
                                    break
                            
                            print(f"Updated Q&A item {item_index}: {new_category} - {new_question[:50]}...")
                            
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not update underlying data: {e}")
                    
                    # Update counters
                    self.update_counters_from_data()
                    
                    edit_window.destroy()
                    messagebox.showinfo("Success", "Changes saved successfully and item marked as accepted!\n\n"
                                    "Note: Changes are in memory only. Use 'Finish and Export' to save permanently.")
                    
            except Exception as e:
                    messagebox.showerror("Error", f"Failed to save changes: ")
                    print(f"Error saving changes: {e}")
        
        def cancel_edit():
            edit_window.destroy()
        
        # Buttons
        ttk.Button(btn_frame, text="Save Changes", command=save_changes, 
           bootstyle=ttk.DARK, width=15).pack(side=LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Cancel", command=cancel_edit, 
           bootstyle=ttk.DARK, width=15).pack(side=LEFT)
        
        # Focus on category entry
        category_entry.focus_set()
        
        # Bind Enter key to save (when not in text widgets)
        def on_enter(event):
            if event.widget not in (question_text, answer_text):
                save_changes()
        
        edit_window.bind('<Return>', on_enter)
        edit_window.bind('<Escape>', lambda e: cancel_edit())
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
        button_frame = Frame(self.left_panel)
        button_frame.pack(pady=10)
        self.accept_btn = ttk.ttk.Button(button_frame, text="Accept", bootstyle=ttk.SUCCESS,
                       width=12,
                        command=self.accept_annotation)
        self.accept_btn.pack(pady=5)

        self.reject_btn = ttk.ttk.Button(button_frame, text="Reject", bootstyle=ttk.DANGER,
                                 width=12, 
                                command=self.reject_annotation)
        self.reject_btn.pack(pady=5)

        self.reset_btn = ttk.ttk.Button(button_frame, text="Reset",bootstyle=ttk.WARNING,
                                width=12,
                                command=self.reset_annotation)
        self.reset_btn.pack(pady=5)
        self.accept_btn.config(state=DISABLED)
        self.reject_btn.config(state=DISABLED)
        self.reset_btn.config(state=DISABLED)
        Label(self.left_panel, text="Base Directory:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.dir_var = StringVar()
        dir_entry = Entry(self.left_panel, textvariable=self.dir_var, width=25)
        dir_entry.pack(pady=5, padx=10)
        
        ttk.Button(self.left_panel, text="Browse Directory", command=self.browse_directory, width=18).pack(pady=5)
        
        Label(self.left_panel, text="Select Video:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        
        listbox_frame = Frame(self.left_panel)
        listbox_frame.pack(pady=5, padx=10, fill=BOTH, expand=True)
        
        self.video_listbox = Listbox(listbox_frame, height=8)
        scrollbar = Scrollbar(listbox_frame, orient=VERTICAL, command=self.video_listbox.yview)
        self.video_listbox.config(yscrollcommand=scrollbar.set)
        self.video_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.video_listbox.bind('<Double-Button-1>', self.on_video_select)
        
        self.load_btn = ttk.Button(self.left_panel, text="Load Selected", command=self.load_selected_video, width=18)
        self.load_btn.pack(pady=5)

    def setup_center_panel(self):
        top_controls = Frame(self.center_panel)
        top_controls.pack(pady=10)
        
        self.switch_btn = ttk.Button(top_controls, text="Switch to Frame Analysis", 
                                command=self.switch_mode,bootstyle=ttk.DARK)
        self.switch_btn.pack(side=LEFT, padx=10)
        
        self.current_video_var = StringVar(value="No video loaded")
        Label(top_controls, textvariable=self.current_video_var, font=("Arial", 10)).pack(side=LEFT, padx=20)
        
        self.frame_label = Label(self.center_panel, width=80, height=30, bg="darkgray", 
                                text="Load a video to begin", font=("Arial", 12))
        self.frame_label.pack(pady=10, expand=True, fill=BOTH)    
          # Video controls
        self.video_controls_frame = Frame(self.center_panel)
        self.video_controls_frame.pack(pady=10)
        ttk.Button(self.video_controls_frame, text="Play/Pause", command=self.toggle_playback, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        ttk.Button(self.video_controls_frame,text="Stop",command=self.stop_video, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        ttk.Button(self.video_controls_frame, text="Fast-Forward", command=self.fast_forward, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        ttk.Button(self.video_controls_frame, text="Rewind", command=self.rewind, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        self.qa_frame = Frame(self.center_panel)
        self.qa_frame.pack(pady=5, padx=10, fill=X)
        self.qa_frame.pack_propagate(False)  # ADD this line to prevent expansion
    
        # Set a fixed height for the Q&A frame
        self.qa_frame.config(height=200)  # ADD this line - adjust height as needed
        
        self.qa_listbox = ttk.Treeview(self.qa_frame, columns=('category', 'question', 'answer', 'status'), 
                                    show='headings', height=6)
        self.qa_listbox.heading('category', text='Category')
        self.qa_listbox.heading('question', text='Question') 
        self.qa_listbox.heading('answer', text='Answer')
        self.qa_listbox.heading('status', text='Status')

        # Configure column widths
        self.qa_listbox.column('category', width=100)
        self.qa_listbox.column('question', width=300)
        self.qa_listbox.column('answer', width=300)
        self.qa_listbox.column('status', width=80)

        self.qa_listbox.pack(side=LEFT, fill=BOTH, expand=True)

        # Add scrollbar
        qa_scrollbar = Scrollbar(self.qa_frame, orient=VERTICAL, command=self.qa_listbox.yview)
        qa_scrollbar.pack(side=RIGHT, fill=Y)
        self.qa_listbox.configure(yscrollcommand=qa_scrollbar.set)

        # Bind events for enhanced functionality
        self.qa_listbox.bind('<Double-1>', self.edit_qa_item)
        self.qa_listbox.bind('<<TreeviewSelect>>', self.on_qa_selection_change)

        # Keep a reference to original pack method for compatibility
        self.qa_listbox.original_pack = self.qa_listbox.pack

        
        # Frame controls (hidden by default)
        self.controls_frame = Frame(self.center_panel)
        ttk.Button(self.controls_frame, text="Previous", command=self.prev_frame, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        self.frame_number_var = StringVar(value="")
        ttk.Label(self.controls_frame, textvariable=self.frame_number_var, bootstyle=ttk.DARK).pack(side=LEFT, padx=10)
        ttk.Button(self.controls_frame, text="Next", command=self.next_frame, bootstyle=ttk.DARK).pack(side=LEFT, padx=5)
        
      

    def setup_right_panel(self):
        Label(self.right_panel, text="CSV File Controls", font=("Arial", 12, "bold"), bg="lightgray").pack(pady=10)
        
        ttk.Button(self.right_panel, text="Split Videos",bootstyle=ttk.PRIMARY,
               width=18, command=self.split_videos).pack(pady=10)
        ttk.Button(self.right_panel, text="Load Existing CSV" ,bootstyle=ttk.PRIMARY,
               width=18, command=self.load_csv_async).pack(pady=5)
        
        Label(self.right_panel, text="Current CSV:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.csv_status_var = StringVar(value="None loaded")
        Label(self.right_panel, textvariable=self.csv_status_var, bg="lightgray", 
              font=("Arial", 8), wraplength=180).pack(pady=5, padx=10)

        
        Label(self.right_panel, text="Export Options:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        ttk.Button(self.right_panel, text="Blur and Track", command=self.blur_track, width=18).pack(pady=5)
        ttk.Button(self.right_panel, text="Question Statistics", command=self.statistics, width=18).pack(pady=5)
        
        Label(self.right_panel, text="Statistics:", bg="lightgray", font=("Arial", 9, "bold")).pack(pady=(20, 5))
        self.current_video_display_var = StringVar(value="Current: No video loaded")
        Label(self.right_panel, textvariable=self.current_video_display_var,bg="lightgray", font=("Arial", 9, "bold"), wraplength=180).pack(pady=(5, 5), padx=5)

       
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
           ttk.Radiobutton(
        self.right_panel,
        text=text,
        variable=self.template_var,           # all buttons share this var
        value=val,                            # value written when selected
    ).pack(anchor="w", padx=20)
        
        ttk.Button(self.right_panel, text="Save Template Selection", 
               command=self.save_chat_template_selection).pack(pady=10)
        ttk.Button(self.right_panel, text="Finish and Export", 
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
        """FIXED: Clean up all video resources properly"""
        print("Cleaning up resources...")
        
        # Stop playback first
        self.playing = False
        
        # Release video capture
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                print("Released video capture")
            except Exception as e:
                print(f"Error releasing video capture: {e}")
        
        # Clear frames if in frame mode
        if self.frames:
            try:
                self.frames.clear()
                gc.collect()
                print("Cleared frames from memory")
            except Exception as e:
                print(f"Error clearing frames: {e}")
        
        # Reset UI state
        self.current_video_path = ""
        self.frame_index = 0
        self.playing = False
        self.auto_play_enabled = True
        self.questions_disabled = True
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
            messagebox.showerror("Error", f"Could not read directory: ")

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
        self.delete_qa_items(0, END)
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
                    self.current_video_display_var.set(f"Current: {video_file}")
                    self.status_var.set(f"Loaded {len(frames)} frames")
                    self.show_qa_for_current_video()
                    self.status_var.set(f"CSV loaded")
                    # ADD THIS: Enable auto-play and disable questions initially
                    if self.mode == "frame":
                        self.auto_play_enabled = False        # no auto-play in this mode
                        self.questions_disabled = False
                        self.enable_question_interaction()
                    else:
                        self.auto_play_enabled = True
                        self.questions_disabled = True
                        self.disable_question_interaction()
                        # Auto-start video playback if video is loaded
                    if self.current_video_path and self.cap is not None:
                        self.playing = True
                        self.update_video()
                        self.status_var.set("Auto-playing video - questions disabled until video completes")
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
            self.root.after(0, lambda: self.status_var.set("Initializing video playbook..."))
            
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
                        self.current_video_display_var.set(f"Current: {video_file}")
                        self.show_qa_for_current_video()
                        
                        # Show first frame immediately
                        ret, frame = self.cap.read()
                        if ret:
                            self._display_bgr_frame(frame)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start
                        
                        # FIXED: Auto-start video playback immediately
                        self.auto_play_enabled = True
                        self.questions_disabled = True
                        self.disable_question_interaction()
                        
                        # Start playing automatically
                        self.playing = True
                        self.update_video()
                        self.status_var.set("Auto-playing video - questions disabled until completion")
                        
                        print("Video auto-playback started successfully")
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

    def update_video(self):
        """FIXED: Update video playback with proper completion handling"""
        if not self.playing or self.cap is None:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Display the frame
                self._display_bgr_frame(frame)
                
                # Schedule next frame (30 FPS = ~33ms delay)
                self.root.after(33, self.update_video)
            else:
                # Video completed - handle end of video
                print("Video playback completed")
                self.playing = False
                
                # If auto-play was enabled, now enable question interaction
                if self.auto_play_enabled and self.questions_disabled:
                    self.enable_question_interaction()
                    self.auto_play_enabled = False
                    self.status_var.set("Video completed - Questions now enabled")
                    messagebox.showinfo("Video Complete", 
                                    "Video playback finished.\nYou can now interact with questions.")
                else:
                    self.status_var.set("Video playback completed")
                
                # Reset video to beginning for replay
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
        except Exception as e:
            print(f"Error in video playback: {e}")
            self.playing = False
            self.status_var.set("Video playback error")
            
            # Enable questions if they were disabled due to auto-play
            if self.auto_play_enabled and self.questions_disabled:
                self.enable_question_interaction()
                self.auto_play_enabled = False
    def toggle_playback(self):
        """FIXED: Toggle play/pause functionality"""
        if self.cap is None or self.is_loading:
            messagebox.showinfo("No Video", "Please load a video first.")
            return
            
        # Disable auto-play mode when user manually controls playback
        if self.auto_play_enabled:
            self.auto_play_enabled = False
            self.enable_question_interaction()
            print("Manual playback control - enabling questions")
        
        self.playing = not self.playing
        if self.playing:
            self.status_var.set("Playing video...")
            self.update_video()
        else:
            self.status_var.set("Video paused")
            print(f"Video {'resumed' if self.playing else 'paused'}")

    def stop_video(self):
        """FIXED: Stop video playback and reset"""
        print("Stopping video playback")
        self.playing = False
        
        if self.cap is not None:
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Show first frame
            ret, frame = self.cap.read()
            if ret:
                self._display_bgr_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position after reading
        
        # Enable questions if they were disabled
        if self.questions_disabled:
            self.enable_question_interaction()
            self.auto_play_enabled = False
        
        self.status_var.set("Video stopped and reset to beginning")

    def fast_forward(self):
        """FIXED: Skip forward 10 seconds in video"""
        if self.cap is None:
            messagebox.showinfo("No Video", "Please load a video first.")
            return
        
        # Disable auto-play when user manually controls
        if self.auto_play_enabled:
            self.auto_play_enabled = False
            self.enable_question_interaction()
        
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        skip_frames = int(1 * fps) if fps > 0 else 300  
        new_frame = min(current_frame + skip_frames, total_frames - 1)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        
        # Show current frame
        ret, frame = self.cap.read()
        if ret:
            self._display_bgr_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)  # Reset position after display
            
            current_time = new_frame / fps if fps > 0 else 0
            self.status_var.set(f"Fast forwarded to {current_time:.1f}s")
            print(f"Fast forwarded to frame {new_frame} ({current_time:.1f}s)")
            
            # Continue playing if it was playing
            if self.playing:
                self.update_video()


    def rewind(self):
        """FIXED: Skip backward 10 seconds in video"""
        if self.cap is None:
            messagebox.showinfo("No Video", "Please load a video first.")
            return
        
        # Disable auto-play when user manually controls
        if self.auto_play_enabled:
            self.auto_play_enabled = False
            self.enable_question_interaction()
        
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        skip_frames = int(1 * fps) if fps > 0 else 300  # 10 seconds worth of frames
        new_frame = max(current_frame - skip_frames, 0)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        
        # Show current frame
        ret, frame = self.cap.read()
        if ret:
            self._display_bgr_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)  # Reset position after display
            
            current_time = new_frame / fps if fps > 0 else 0
            self.status_var.set(f"Rewound to {current_time:.1f}s")
            print(f"Rewound to frame {new_frame} ({current_time:.1f}s)")
            
            # Continue playing if it was playing
            if self.playing:
                self.update_video()
    def _show_video_controls(self):
        self.video_controls_frame.pack_forget()              
        # keep it right above the Q&A table
        self.video_controls_frame.pack(pady=10, before=self.qa_frame)
    def switch_mode(self):
        """Switch between video and frame analysis modes"""
        if self.is_loading:
            messagebox.showinfo("Please Wait", "Cannot switch modes while loading.")
            return
            
        if self.mode == "video":
            self.mode = "frame"
            self.switch_btn.config(text="Switch to Video Playback")
            self.video_controls_frame.pack_forget()
            self.controls_frame.pack(pady=10, before=self.qa_frame)
            print("Switched to frame mode")
        else:
            self.mode = "video"
            self.switch_btn.config(text="Switch to Frame Analysis")
            self.controls_frame.pack_forget()
            self._show_video_controls()
            print("Switched to video mode")
        
        # Reload current video in new mode if available
        if self.current_video_path:
            self.load_selected_video()

    # CSV Methods
    def load_csv_async(self):
        """Load CSV file asynchronously with status column support"""
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
                        # Add status field if it doesn't exist
                        if 'status' not in row:
                            row['status'] = 'none'
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
                    messagebox.showerror("Error", f"Could not load CSV file: ")
                
                self.root.after(0, show_error)
        
        threading.Thread(target=load_csv_thread, daemon=True).start()


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
        """Enhanced accept with status persistence to data"""
        if self.questions_disabled:
            return
            
        selection = self.get_qa_selection()
        if not selection:
            messagebox.showinfo("Accept", "No question selected.")
            return
        
        if hasattr(self.qa_listbox, 'selection'):  # Treeview mode
            item_id = selection[0]
            
            # Save current state to history for undo
            self.qa_history.append({
                'action': 'accept',
                'item_id': item_id,
                'previous_state': self.qa_states.get(item_id, 'none'),
                'timestamp': time.time()
            })
            
            # Update UI state and appearance
            self.qa_states[item_id] = 'accepted'
            values = self.qa_listbox.item(item_id, 'values')
            self.qa_listbox.item(item_id, values=(*values[:3], 'Accepted'))
            
            # Color the row green
            self.qa_listbox.item(item_id, tags=('accepted',))
            self.qa_listbox.tag_configure('accepted', background='lightgreen')
            
            # UPDATE DATA: Find corresponding item and update status
            try:
                all_items = self.qa_listbox.get_children()
                item_index = list(all_items).index(item_id)
                qa_master   = self.current_video_qa[item_index]    # same dict object!
                qa_master['status'] = 'accepted'   
                
                if 0 <= item_index < len(self.current_video_qa):
                    # Update in current video data
                    self.current_video_qa[item_index]['status'] = 'accepted'
                    
                    # Update in main qa_data
                    video_file_name = os.path.basename(self.current_video_path) if self.current_video_path else ""
                    for qa in self.qa_data:
                        video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                                    qa.get("video_path") or qa.get("file_name") or "")
                        if (video_file == video_file_name and 
                            qa.get('question') == values[1] and  
                            qa.get('answer') == values[2]):
                            qa['status'] = 'accepted'
                            break
                            
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not update data status: {e}")
            
            # Update counters and button states
            self.update_counters_from_data()
            self.update_button_states()
            
            # messagebox.showinfo("Accept", "Question accepted successfully")
            
        else:  
            newly_accepted = []
            for i in selection:
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


    def reject_annotation(self):
        """Enhanced reject with status persistence to data"""
        if self.questions_disabled:
            return
            
        selection = self.get_qa_selection()
        if not selection:
            messagebox.showinfo("Reject", "No question selected.")
            return
        
        if hasattr(self.qa_listbox, 'selection'):  # Treeview mode
            item_id = selection[0]
            
            # Save current state to history for undo
            self.qa_history.append({
                'action': 'reject',
                'item_id': item_id,
                'previous_state': self.qa_states.get(item_id, 'none'),
                'timestamp': time.time()
            })
            
            # Update UI state and appearance
            self.qa_states[item_id] = 'rejected'
            values = self.qa_listbox.item(item_id, 'values')
            self.qa_listbox.item(item_id, values=(*values[:3], 'Rejected'))

            # Color the row red
            self.qa_listbox.item(item_id, tags=('rejected',))
            self.qa_listbox.tag_configure('rejected', background='lightcoral')
            

            # Make item editable
            self.editable_questions.add(item_id)
            
            # UPDATE DATA: Find corresponding item and update status
            try:
                all_items = self.qa_listbox.get_children()
                item_index = list(all_items).index(item_id)
                qa_master   = self.current_video_qa[item_index]    # same dict object!
                qa_master['status'] = 'rejected'      
                if 0 <= item_index < len(self.current_video_qa):
                    # Update in current video data
                    self.current_video_qa[item_index]['status'] = 'rejected'
                    
                    # Update in main qa_data
                    video_file_name = os.path.basename(self.current_video_path) if self.current_video_path else ""
                    for qa in self.qa_data:
                        video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                                    qa.get("video_path") or qa.get("file_name") or "")
                        if (video_file == video_file_name and 
                            qa.get('question') == values[1] and  
                            qa.get('answer') == values[2]):
                            qa['status'] = 'rejected'
                            break
                            
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not update data status: {e}")
            
            # Update counters and button states
            self.update_counters_from_data()
            self.update_button_states()
            
            # messagebox.showinfo("Reject", "Question rejected. Double-click to edit.")

    def reset_annotation(self):
        """Reset the last action (undo functionality) with data persistence"""
        if not self.qa_history:
            messagebox.showinfo("Reset", "No actions to undo")
            return
        
        # Get last action
        last_action = self.qa_history.pop()
        item_id = last_action['item_id']
        previous_state = last_action['previous_state']
        
        # Restore previous state
        self.qa_states[item_id] = previous_state
        
        # Update appearance
        if previous_state == 'none':
            self.qa_listbox.item(item_id, values=(*self.qa_listbox.item(item_id, 'values')[:3], 'None'))
            self.qa_listbox.item(item_id, tags=())
        elif previous_state == 'accepted':
            self.qa_listbox.item(item_id, values=(*self.qa_listbox.item(item_id, 'values')[:3], 'Accepted'))
            self.qa_listbox.item(item_id, tags=('accepted',))
        elif previous_state == 'rejected':
            self.qa_listbox.item(item_id, values=(*self.qa_listbox.item(item_id, 'values')[:3], 'Rejected'))
            self.qa_listbox.item(item_id, tags=('rejected',))
        
        # UPDATE DATA: Find corresponding item and update status
        try:
            all_items = self.qa_listbox.get_children()
            item_index = list(all_items).index(item_id)
            values = self.qa_listbox.item(item_id, 'values')
            qa_master = self.current_video_qa[item_index]
            qa_master['status'] = previous_state    # restore
            
            if 0 <= item_index < len(self.current_video_qa):
                # Update in current video data
                self.current_video_qa[item_index]['status'] = previous_state
                
                # Update in main qa_data
                video_file_name = os.path.basename(self.current_video_path) if self.current_video_path else ""
                for qa in self.qa_data:
                    video_file = (qa.get("video_file_path") or qa.get("video_file") or 
                                qa.get("video_path") or qa.get("file_name") or "")
                    if (video_file == video_file_name and 
                        qa.get('question') == values[1] and  
                        qa.get('answer') == values[2]):
                        qa['status'] = previous_state
                        break
                        
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not update data status: {e}")
        
        # Remove from editable if needed
        if previous_state != 'rejected':
            self.editable_questions.discard(item_id)
        
        # Update counters and button states
        self.update_counters_from_data()
        self.update_button_states()
        
        messagebox.showinfo("Reset", f"Undid {last_action['action']} action")


    def update_counters_from_data(self):
        """Update counters based on actual data status instead of UI state"""
        if not self.current_video_qa:
            self.accepted_var.set("0")
            self.rejected_var.set("0")
            return
        
        accepted_count = sum(1 for qa in self.current_video_qa if qa.get('status') == 'accepted')
        rejected_count = sum(1 for qa in self.current_video_qa if qa.get('status') == 'rejected')
        
        self.accepted_var.set(str(accepted_count))
        self.rejected_var.set(str(rejected_count))
    

    def write_qa_to_csv(self, path, qa_list):
        """Write QA data to CSV file with status column"""
        fieldnames = ["index", "video_file_path", "question", "category", "answer", "status"]
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qa in qa_list:
                clean_row = {k: qa.get(k, "") for k in fieldnames}
                # Ensure status has a default value
                if not clean_row.get('status'):
                    clean_row['status'] = 'none'
                writer.writerow(clean_row)

    def append_qa_to_csv(self, path, qa_list):
        """Append QA data to CSV file with status column"""
        if not qa_list:
            return
            
        fieldnames = ["index", "video_file_path", "question", "category", "answer", "status"]
        write_header = not os.path.exists(path)
        
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for qa in qa_list:
                clean_row = {k: qa.get(k, "") for k in fieldnames}
                # Ensure status has a default value
                if not clean_row.get('status'):
                    clean_row['status'] = 'none'
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
            messagebox.showerror("Error", f"Could not process videos: ")

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
            messagebox.showerror("Error", "Failed to start blur and track: ")

    def start_blur_and_track_process_multiple(self, circles):
        """Start the blur and track process for multiple circles in a separate thread"""
        self.status_var.set("Starting SAM blur and track for multiple objects...")
        
        def blur_track_thread():
            try:
                self.blur_and_track_yolo_sam_predictive(circles)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Blur and track failed: "))
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
        if qa.get('status') == 'accepted'
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
                        self.current_video_display_var.set(f"Current: {current_video_name}")
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
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate templates: "))
            
            self.status_var.set("Generating templates...")
            threading.Thread(target=generate_templates, daemon=True).start()

    def finish_and_export_chat_template(self):
            """Export chat templates"""
             # Use qa_data with status filtering instead of accepted_qa_data

            accepted_qa_data = [ qa for qa in self.current_video_qa if qa.get('status') == 'accepted']
            if not accepted_qa_data:

          
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

             
                """
                Build the prompt JSONL files from every row whose
                status == "accepted" and save them to *out_dir*.
                Also updates the CSV (backup first) with the latest statuses.
                """

                try:
                    # visual feedback
                    self.root.after(0, lambda: self.status_var.set("Exporting templates..."))

                    # 1) gather the rows that were accepted
                    accepted_rows   = [qa for qa in self.qa_data if qa.get("status") == "accepted"]
                    processed_count = 0
                    skipped_count   = 0
                    prompts         = []                           # (template-id, prompt-dict)

                    for qa in accepted_rows:
                        video_file = (qa.get("video_file_path") or qa.get("video_file") or
                                    qa.get("video_path")      or qa.get("file_name")   or "")
                        vpath = os.path.join(self.video_dir, video_file) if video_file else ""
                        q     = qa.get("question", "")
                        a     = qa.get("answer",  "")

                        if not (vpath and q and a):
                            skipped_count += 1
                            continue

                        if template_choice == "4":                # export *all* templates
                            for tid in ("1", "2", "3"):
                                prompts.append((tid,
                                                self.PROMPT_BUILDERS[tid](vpath, q, a, num_frames=4)))
                        else:                                     # only the chosen one
                            prompts.append((template_choice,
                                            self.PROMPT_BUILDERS[template_choice](vpath, q, a, num_frames=4)))

                        processed_count += 1

                    if not prompts:
                        raise RuntimeError("No accepted prompts to export.")

                    # 2) back-up the CSV, then write the updated version
                    if self.csv_file_path:
                        ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = self.csv_file_path.replace(".csv", f"_backup_{ts}.csv")
                        shutil.copy2(self.csv_file_path, backup_path)

                        self.write_qa_to_csv(self.csv_file_path, self.qa_data)
                        print(f"CSV updated; backup saved to {backup_path}")

                    # 3) write the prompt file(s)
                    if template_choice == "4":
                        groups = {"1": [], "2": [], "3": []}
                        for tid, data in prompts:
                            groups[tid].append(data)

                        out_files = []
                        for tid, plist in groups.items():
                            if not plist:
                                continue
                            fname = f"model_train_{self.FILE_SUFFIX[tid]}.jsonl"
                            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                                for obj in plist:
                                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            out_files.append(fname)
                    else:
                        fname = f"model_train_{self.FILE_SUFFIX[template_choice]}.jsonl"
                        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                            for _, obj in prompts:
                                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        out_files = [fname]

                    # 4) UI success message
                    def done():
                        msg = (f"Export complete!\n\nProcessed: {processed_count}\n"
                            f"Skipped: {skipped_count}\n\n"
                            "Files written:\n  " + "\n  ".join(out_files))
                        messagebox.showinfo("Export Complete", msg)
                        self.status_var.set("Export completed")
                    self.root.after(0, done)

                    # clear in-memory prompt cache
                    self.saved_prompts.clear()

                except Exception as e:
                    print(e)
                    def fail():
                        messagebox.showerror("Export Error","Error")
                        self.status_var.set("Export failed")
                    self.root.after(0, fail)

            threading.Thread(target=export_templates, daemon=True).start()


             
    def cleanup_on_exit(self):
            """Cleanup when application exits"""
            print("Application shutting down...")
            self.stop_current_thread()
            self.cleanup_resources()
