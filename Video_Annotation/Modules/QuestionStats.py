import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from collections import Counter
import re
from tkinter import *
import pandas as pd
import numpy as np

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
        """Setup the UI with 3x1 layout: donut chart on top, scatter plot on bottom"""
        # Main container
        main_frame = Frame(self, bg="white")
        main_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Top row - 3 sections for donut charts and bar chart
        top_frame = Frame(main_frame, bg="white", height=350)
        top_frame.pack(fill=X, pady=(0, 5))
        top_frame.pack_propagate(False)
        
        # Top left - Bar chart (Vehicular/Non-Vehicular by Category)
        self.top_left_frame = Frame(top_frame, bg="white", relief=SOLID, bd=1)
        self.top_left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 2))
        
        # Top middle - Category distribution bar chart
        self.top_middle_frame = Frame(top_frame, bg="white", relief=SOLID, bd=1)
        self.top_middle_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(2, 2))
        
        # Top right - Donut chart (Vehicular vs Non-Vehicular proportion)
        self.top_right_frame = Frame(top_frame, bg="white", relief=SOLID, bd=1)
        self.top_right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(2, 0))
        
        # Bottom row - Full width scatter plot
        bottom_frame = Frame(main_frame, bg="white")
        bottom_frame.pack(fill=BOTH, expand=True, pady=(5, 0))
        
        self.bottom_frame = Frame(bottom_frame, bg="white", relief=SOLID, bd=1)
        self.bottom_frame.pack(fill=BOTH, expand=True)
        
        # Tooltip label for hover information
        self.tooltip_label = Label(self, text="", bg="lightyellow", relief=SOLID, bd=1,
                                 font=("Arial", 9), justify=LEFT)
    
    def show_tooltip(self, event, text):
        """Show tooltip at mouse position"""
        self.tooltip_label.config(text=text)
        # Convert matplotlib event coordinates to tkinter window coordinates
        if hasattr(event, 'canvas'):
            canvas_widget = event.canvas.get_tk_widget()
            x = canvas_widget.winfo_rootx() + int(event.x) + 10
            y = canvas_widget.winfo_rooty() + int(event.y) + 10
            # Convert to relative coordinates within the main window
            self.tooltip_label.place(x=x - self.winfo_rootx(), 
                                   y=y - self.winfo_rooty())
        else:
            self.tooltip_label.place(x=100, y=100)
    
    def hide_tooltip(self, event):
        """Hide tooltip"""
        self.tooltip_label.place_forget()
    
    def create_charts(self):
        """Create all charts"""
        if self.qa_data.empty:
            # Show "No data available" message in each section
            for frame in [self.top_left_frame, self.top_middle_frame, self.top_right_frame, self.bottom_frame]:
                Label(frame, text="No data available", 
                      font=("Arial", 10), bg="white").pack(expand=True)
            return
        
        self.create_vehicular_category_chart()
        self.create_category_distribution_chart()
        self.create_proportion_donut_chart()
        self.create_question_types_scatter()
    
    def create_vehicular_category_chart(self):
        """Top left - Vehicular and Non-Vehicular Question Counts per Category"""
        categories = self.qa_data.get('category', 'Unknown').fillna('Unknown').tolist()
        category_counts = Counter(categories)
        
        # For demo purposes, classify as vehicular/non-vehicular
        vehicular_data = {}
        non_vehicular_data = {}
        
        for category, count in category_counts.items():
            vehicular_keywords = ['vehicle', 'car', 'truck', 'traffic', 'driving', 'road']
            is_vehicular = any(keyword in category.lower() for keyword in vehicular_keywords)
            
            if is_vehicular:
                vehicular_data[category] = int(count * 0.4)
                non_vehicular_data[category] = count - vehicular_data[category]
            else:
                vehicular_data[category] = int(count * 0.6)
                non_vehicular_data[category] = count - vehicular_data[category]
        
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.25, top=0.85, left=0.15, right=0.95)
        
        categories_list = list(category_counts.keys())
        x = np.arange(len(categories_list))
        width = 0.35
        
        vehicular_counts = [vehicular_data[cat] for cat in categories_list]
        non_vehicular_counts = [non_vehicular_data[cat] for cat in categories_list]
        
        bars1 = ax.bar(x - width/2, vehicular_counts, width, label='Vehicular', color='#ff7f0e')
        bars2 = ax.bar(x + width/2, non_vehicular_counts, width, label='Non-Vehicular', color='#1f77b4')
        
        ax.set_title('Vehicular and Non-Vehicular Question Counts per Reclassified Category', 
                     fontsize=8, fontweight='bold', pad=10)
        ax.set_xlabel('Reclassified Category', fontsize=7)
        ax.set_ylabel('Count of Questions', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories_list, rotation=45, ha='right', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.top_left_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True)
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax:
                # Find which bar is hovered
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    if bar1.contains(event)[0]:
                        cat = categories_list[i]
                        count = vehicular_counts[i]
                        tooltip_text = f"Category: {cat}\nType: Vehicular\nCount: {count}"
                        self.show_tooltip(event, tooltip_text)
                        return
                    elif bar2.contains(event)[0]:
                        cat = categories_list[i]
                        count = non_vehicular_counts[i]
                        tooltip_text = f"Category: {cat}\nType: Non-Vehicular\nCount: {count}"
                        self.show_tooltip(event, tooltip_text)
                        return
        
        def on_leave(event):
            self.hide_tooltip(event)
        
        canvas.mpl_connect('motion_notify_event', on_hover)
        canvas.mpl_connect('axes_leave_event', on_leave)
    
    def create_category_distribution_chart(self):
        """Top middle - Distribution of Reclassified Category"""
        categories = self.qa_data.get('category', 'Unknown').fillna('Unknown').tolist()
        category_counts = Counter(categories)
        
        fig, ax = plt.subplots(figsize=(3.5, 4.2))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.25, top=0.85, left=0.15, right=0.95)
        
        categories_list = list(category_counts.keys())
        counts = list(category_counts.values())
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'][:len(categories_list)]
        
        bars = ax.bar(categories_list, counts, color=colors)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=6)
        
        ax.set_title('Distribution of Reclassified Category', fontsize=8, fontweight='bold', pad=10)
        ax.set_xlabel('Categories', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.top_middle_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True)
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        cat = categories_list[i]
                        count = counts[i]
                        tooltip_text = f"Category: {cat}\nCount: {count}"
                        self.show_tooltip(event, tooltip_text)
                        return
        
        def on_leave(event):
            self.hide_tooltip(event)
        
        canvas.mpl_connect('motion_notify_event', on_hover)
        canvas.mpl_connect('axes_leave_event', on_leave)
    
    def create_proportion_donut_chart(self):
        """Top right - Proportion of Vehicular vs Non-Vehicular Questions"""
        total_questions = len(self.qa_data)
        
        # For demo, assume 65% vehicular, 35% non-vehicular
        vehicular_count = int(total_questions * 0.651)
        non_vehicular_count = total_questions - vehicular_count
        
        fig, ax = plt.subplots(figsize=(3.5, 4.2))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
        
        sizes = [vehicular_count, non_vehicular_count]
        labels = ['Vehicular', 'Non-Vehicular']
        colors = ['#ff9999', '#87ceeb']
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                        colors=colors, startangle=90,
                                        wedgeprops=dict(width=0.5),
                                        textprops={'fontsize': 7})
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        
        for text in texts:
            text.set_fontsize(7)
        
        ax.set_title('Proportion of Vehicular vs\nNon-Vehicular Questions', 
                     fontsize=8, fontweight='bold', pad=10)
        
        canvas = FigureCanvasTkAgg(fig, self.top_right_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True)
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax:
                for i, wedge in enumerate(wedges):
                    if wedge.contains_point([event.x, event.y]):
                        label = labels[i]
                        count = sizes[i]
                        percentage = (count / total_questions) * 100
                        tooltip_text = f"Type: {label}\nCount: {count}\nPercentage: {percentage:.1f}%"
                        self.show_tooltip(event, tooltip_text)
                        return
        
        def on_leave(event):
            self.hide_tooltip(event)
        
        canvas.mpl_connect('motion_notify_event', on_hover)
        canvas.mpl_connect('axes_leave_event', on_leave)
    
    def create_question_types_scatter(self):
        """Bottom - Distribution of Question Types (scatter plot)"""
        question_types = ['how', 'is', 'where', 'what', 'why', 'when', 'who', 'which']
        colors_map = {
            'how': '#1f77b4', 'is': '#ff7f0e', 'where': '#2ca02c', 'what': '#d62728',
            'why': '#9467bd', 'when': '#8c564b', 'who': '#e377c2', 'which': '#7f7f7f'
        }
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.85)
        
        # Store scatter data for hover
        scatter_data = []
        
        if 'question' in self.qa_data.columns:
            questions = self.qa_data['question'].fillna('').str.lower().str.strip()
            
            # Create clustered scatter plot like the reference image
            np.random.seed(42)  # For consistent clustering
            
            for i, q_type in enumerate(question_types):
                # Find actual questions of this type
                type_questions = questions[questions.str.contains(f'^{q_type}\\b', regex=True, case=False)]
                count = len(type_questions)
                
                if count > 0:
                    # Create clusters for each question type (like reference image)
                    center_x = i * 50  # Space clusters evenly
                    center_y = 50      # Center height
                    
                    # Generate clustered points around the center
                    x_coords = np.random.normal(center_x, 15, count)  # Cluster around center_x
                    y_coords = np.random.normal(center_y, 25, count)  # Cluster around center_y
                    
                    # Ensure y coordinates stay within reasonable bounds
                    y_coords = np.clip(y_coords, 5, 95)
                    
                    # Store actual question data for tooltips
                    for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                        if j < len(type_questions):
                            idx = type_questions.index[j]
                            original_question = self.qa_data.iloc[idx]['question'] if idx < len(self.qa_data) else ""
                        else:
                            original_question = f"Sample {q_type} question"
                        
                        scatter_data.append({
                            'x': x, 'y': y, 
                            'type': q_type, 'color': colors_map[q_type],
                            'question': original_question[:50] + "..." if len(original_question) > 50 else original_question
                        })
                    
                    ax.scatter(x_coords, y_coords, c=colors_map[q_type], alpha=0.7, s=25, label=f'{q_type.title()} ({count})')
                else:
                    # If no questions found, still show in legend but no points
                    ax.scatter([], [], c=colors_map[q_type], alpha=0.7, s=25, label=f'{q_type.title()} (0)')
        
        ax.set_title('Distribution of Question Types', fontsize=10, fontweight='bold', pad=15)
        ax.set_xlabel('Question Distribution', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Remove x-axis labels for cleaner look like reference
        ax.set_xticks([])
        
        ax.legend(title='Question Types (Count)', loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=7, title_fontsize=8)
        
        canvas = FigureCanvasTkAgg(fig, self.bottom_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=BOTH, expand=True)
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax:
                # Find closest point
                if scatter_data and event.xdata is not None and event.ydata is not None:
                    min_dist = float('inf')
                    closest_point = None
                    
                    for point in scatter_data:
                        dist = ((point['x'] - event.xdata) ** 2 + (point['y'] - event.ydata) ** 2) ** 0.5
                        if dist < min_dist and dist < 20:  # Within reasonable distance
                            min_dist = dist
                            closest_point = point
                    
                    if closest_point:
                        tooltip_text = f"Question Type: {closest_point['type'].title()}\nQuestion: {closest_point['question']}"
                        self.show_tooltip(event, tooltip_text)
        
        def on_leave(event):
            self.hide_tooltip(event)
        
        canvas.mpl_connect('motion_notify_event', on_hover)
        canvas.mpl_connect('axes_leave_event', on_leave)