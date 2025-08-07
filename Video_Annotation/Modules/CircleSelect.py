import cv2
import ttkbootstrap as ttk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from segment_anything import sam_model_registry, SamPredictor



SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

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
        
        self.process_btn = ttk.Button(button_frame, text="Process Video", command=self.confirm,
                               bootstyle=ttk.SUCCESS, width=15)
        self.process_btn.pack(side=LEFT, padx=10)
        self.process_btn.config(state=DISABLED)  # Disabled until circle is drawn
        
        self.clear_btn = ttk.Button(button_frame, text="Clear All Circles", command=self.clear_all_circles,
                             bootstyle=ttk.WARNING, width=15)
        self.clear_btn.pack(side=LEFT, padx=10)
        self.clear_btn.config(state=DISABLED)
        
        self.undo_btn = ttk.Button(button_frame, text="Undo Last", command=self.undo_last_circle,
                             bootstyle=ttk.INFO, width=15)
        self.undo_btn.pack(side=LEFT, padx=10)
        self.undo_btn.config(state=DISABLED)
        
        self.cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.destroy,
                              bootstyle=ttk.DANGER, width=15)
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