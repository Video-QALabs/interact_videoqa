import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import Counter
import re
from tkinter import *
import pandas as pd
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
