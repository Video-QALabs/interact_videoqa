from tkinter import Tk
from Modules.AsyncVideoAnnotator import AsyncVideoAnnotationTool



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