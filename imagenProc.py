import nibabel as nib
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
from PIL import Image, ImageTk
import cv2
import os
import sys
import json

class NiftiViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("NIfTI Viewer with 3D Visualization and Drawing")
        self.root.geometry("800x700")
        
        # Variables
        self.image_data = None
        self.nii_image = None
        self.width, self.height, self.depth = 0, 0, 0
        self.corte_actual = "Axial"
        self.indice_corte = 0
        self.file_path = None
        
        # Drawing variables
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.draw_radius = 3
        self.draw_color = (255, 0, 0)  # Red by default
        self.draw_points = []  # List to store (x, y, z, slice_type) of drawn points
        self.overlay_data = None  # 3D array for drawn overlay
        self.current_display_img = None  # Store current displayed image
        self.seed_selection_mode = False
        
        # UI Elements
        self.create_ui()
        
        # Set window icon
        icon_path = self.resource_path("brain_icon.png")
        if os.path.exists(icon_path):
            try:
                icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, icon)
            except Exception as e:
                print(f"Error loading icon: {e}")
    
    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_path, relative_path)
        except Exception:
            return relative_path
    
    def create_ui(self):
        # Create menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open NIfTI File", command=self.load_image)
        filemenu.add_separator()
        filemenu.add_command(label="Save Drawings", command=self.save_drawings, state="disabled")
        filemenu.add_command(label="Load Drawings", command=self.load_drawings, state="disabled")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Axial View", command=lambda: self.change_slice_type("Axial"), state="disabled")
        viewmenu.add_command(label="Sagittal View", command=lambda: self.change_slice_type("Sagittal"), state="disabled")
        viewmenu.add_command(label="Coronal View", command=lambda: self.change_slice_type("Coronal"), state="disabled")
        viewmenu.add_separator()
        viewmenu.add_command(label="3D Visualization", command=self.visualize_3d, state="disabled")
        menubar.add_cascade(label="View", menu=viewmenu)
        
        drawmenu = tk.Menu(menubar, tearoff=0)
        drawmenu.add_command(label="Change Draw Color", command=self.choose_color, state="disabled")
        drawmenu.add_command(label="Clear Drawings", command=self.clear_drawings, state="disabled")
        drawmenu.add_separator()
        
        # Submenu for brush size
        sizemenu = tk.Menu(drawmenu, tearoff=0)
        sizemenu.add_command(label="Small (1px)", command=lambda: self.set_brush_size(1), state="disabled")
        sizemenu.add_command(label="Medium (3px)", command=lambda: self.set_brush_size(3), state="disabled")
        sizemenu.add_command(label="Large (5px)", command=lambda: self.set_brush_size(5), state="disabled")
        drawmenu.add_cascade(label="Brush Size", menu=sizemenu)
        
        menubar.add_cascade(label="Drawing", menu=drawmenu)

        segmenu = tk.Menu(menubar, tearoff=0)
        segmenu.add_command(label="Umbralización", command=lambda: self.show_segmentation_options("Umbralización"), state="disabled")
        segmenu.add_command(label="Crecimiento de Regiones", command=lambda: self.show_segmentation_options("Crecimiento"), state="disabled")
        segmenu.add_command(label="K-Means", command=lambda: self.show_segmentation_options("K-Means"), state="disabled")
        menubar.add_cascade(label="Segmentación", menu=segmenu)

        self.segmenu = segmenu

        prepmenu = tk.Menu(menubar, tearoff=0)
        prepmenu.add_command(label="Filtro Media", command=lambda: self.show_preprocessing_options("Media"), state="disabled")
        prepmenu.add_command(label="Filtro Mediana", command=lambda: self.show_preprocessing_options("Mediana"), state="disabled")
        prepmenu.add_command(label="Filtro Bilateral (preserva bordes)", command=lambda: self.show_preprocessing_options("Bilateral"), state="disabled")
        prepmenu.add_command(label="Filtro Anisotrópico (preserva bordes)", command=lambda: self.show_preprocessing_options("Anisotropico"), state="disabled")
        prepmenu.add_command(label="Filtro Canny (Detección de Bordes)", command=lambda: self.show_preprocessing_options("Bordes"), state="disabled")
        prepmenu.add_command(label="Non-local Means", command=lambda: self.show_preprocessing_options("NLM"), state="disabled")
        prepmenu.add_command(label="Roberts Edge Detection", command=lambda: self.show_preprocessing_options("Roberts"), state="disabled")
        prepmenu.add_command(label="Laplacian of Gaussian (LoG)", command=lambda: self.show_preprocessing_options("LoG"), state="disabled")
        menubar.add_cascade(label="Preprocesamiento", menu=prepmenu)

        self.prepmenu = prepmenu
        
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        self.root.config(menu=menubar)
        self.viewmenu = viewmenu
        self.drawmenu = drawmenu
        self.sizemenu = sizemenu
        self.filemenu = filemenu
        
        # Frame for controls
        control_frame = ttk.LabelFrame(self.root, text="Controls")
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Buttons for file selection
        self.btn_load = ttk.Button(control_frame, text="Open NIfTI File", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=5, pady=5)
        
        # File info label
        self.label_info = ttk.Label(control_frame, text="No file loaded")
        self.label_info.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Frame for view selection
        view_frame = ttk.LabelFrame(control_frame, text="View Selection")
        view_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.btn_axial = ttk.Button(view_frame, text="Axial", command=lambda: self.change_slice_type("Axial"), state="disabled")
        self.btn_axial.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_sagittal = ttk.Button(view_frame, text="Sagittal", command=lambda: self.change_slice_type("Sagittal"), state="disabled")
        self.btn_sagittal.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_coronal = ttk.Button(view_frame, text="Coronal", command=lambda: self.change_slice_type("Coronal"), state="disabled")
        self.btn_coronal.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_3d = ttk.Button(view_frame, text="3D View", command=self.visualize_3d, state="disabled")
        self.btn_3d.grid(row=0, column=3, padx=5, pady=5)
        
        # Drawing controls
        draw_frame = ttk.LabelFrame(control_frame, text="Drawing Tools")
        draw_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.btn_color = ttk.Button(draw_frame, text="Color", command=self.choose_color, state="disabled")
        self.btn_color.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_clear = ttk.Button(draw_frame, text="Clear", command=self.clear_drawings, state="disabled")
        self.btn_clear.grid(row=0, column=1, padx=5, pady=5)
        
        self.draw_mode_var = tk.BooleanVar()
        self.draw_mode_var.set(False)
        self.chk_draw = ttk.Checkbutton(draw_frame, text="Draw Mode", variable=self.draw_mode_var, state="disabled")
        self.chk_draw.grid(row=0, column=2, padx=5, pady=5)
        
        self.color_preview = tk.Canvas(draw_frame, width=20, height=20, bg="red")
        self.color_preview.grid(row=0, column=3, padx=5, pady=5)
        
        # Frame for slice navigation
        slice_frame = ttk.LabelFrame(self.root, text="Slice Navigation")
        slice_frame.pack(fill="x", padx=10, pady=5)
        
        self.slice_slider = ttk.Scale(slice_frame, from_=0, to=100, orient="horizontal", command=self.update_slice)
        self.slice_slider.pack(fill="x", padx=10, pady=5)
        self.slice_slider.state(["disabled"])
        
        self.slice_label = ttk.Label(slice_frame, text="Slice: 0/0")
        self.slice_label.pack(pady=2)
        
        # Frame for image display
        display_frame = ttk.LabelFrame(self.root, text="Image Display")
        display_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(display_frame)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        # Coordinate display
        self.coord_var = tk.StringVar()
        self.coord_var.set("Coordinates: -")
        self.coord_label = ttk.Label(self.root, textvariable=self.coord_var, relief="sunken", anchor="e")
        self.coord_label.pack(side="bottom", fill="x")
    
    def load_image(self):
        """Load a NIfTI image file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Select NIfTI Image File"
            )
            
            if not file_path:
                return
            
            self.status_var.set("Loading image...")
            self.root.update_idletasks()
            
            self.file_path = file_path
            self.nii_image = nib.load(file_path)
            self.image_data = self.nii_image.get_fdata()
            
            # Get dimensions
            self.width, self.height, self.depth = self.image_data.shape
            
            # Initialize overlay data
            self.overlay_data = np.zeros_like(self.image_data)
            
            # Clear stored drawn points
            self.draw_points = []
            
            # Update UI
            filename = os.path.basename(file_path)
            self.label_info.config(text=f"File: {filename}\nDimensions: {self.width}×{self.height}×{self.depth}")
            
            # Enable controls
            self.btn_axial.state(["!disabled"])
            self.btn_sagittal.state(["!disabled"])
            self.btn_coronal.state(["!disabled"])
            self.btn_3d.state(["!disabled"])
            self.slice_slider.state(["!disabled"])
            self.btn_color.state(["!disabled"])
            self.btn_clear.state(["!disabled"])
            self.chk_draw.state(["!disabled"])
            
            # Enable menu items
            self.viewmenu.entryconfig("Axial View", state="normal")
            self.viewmenu.entryconfig("Sagittal View", state="normal")
            self.viewmenu.entryconfig("Coronal View", state="normal")
            self.viewmenu.entryconfig("3D Visualization", state="normal")
            self.segmenu.entryconfig("Umbralización", state="normal")
            self.segmenu.entryconfig("Crecimiento de Regiones", state="normal")
            self.segmenu.entryconfig("K-Means", state="normal")
            self.prepmenu.entryconfig("Filtro Media", state="normal")
            self.prepmenu.entryconfig("Filtro Mediana", state="normal")
            self.prepmenu.entryconfig("Filtro Bilateral (preserva bordes)", state="normal")
            self.prepmenu.entryconfig("Filtro Anisotrópico (preserva bordes)", state="normal")
            self.prepmenu.entryconfig("Filtro Canny (Detección de Bordes)", state="normal")
            self.prepmenu.entryconfig("Non-local Means", state="normal")
            self.prepmenu.entryconfig("Roberts Edge Detection", state="normal")
            self.prepmenu.entryconfig("Laplacian of Gaussian (LoG)", state="normal")

            # Enable drawing menu items
            self.drawmenu.entryconfig("Change Draw Color", state="normal")
            self.drawmenu.entryconfig("Clear Drawings", state="normal")
            self.sizemenu.entryconfig("Small (1px)", state="normal")
            self.sizemenu.entryconfig("Medium (3px)", state="normal")
            self.sizemenu.entryconfig("Large (5px)", state="normal")
            self.filemenu.entryconfig("Save Drawings", state="normal")
            self.filemenu.entryconfig("Load Drawings", state="normal")
            
            # Set default view
            self.change_slice_type("Axial")
            self.status_var.set(f"Loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def change_slice_type(self, slice_type):
        """Change the slice orientation"""
        if self.image_data is None:
            return
        
        self.corte_actual = slice_type
        
        # Update slider range based on view
        if slice_type == "Axial":
            max_slice = self.depth - 1
            self.indice_corte = self.depth // 2
        elif slice_type == "Sagittal":
            max_slice = self.width - 1
            self.indice_corte = self.width // 2
        else:  # Coronal
            max_slice = self.height - 1
            self.indice_corte = self.height // 2
        
        self.slice_slider.config(from_=0, to=max_slice)
        self.slice_slider.set(self.indice_corte)
        self.update_slice()
    
    def normalize_image(self, img):
        """Normalize image to 0-255 range"""
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    def apply_colormap(self, img):
        """Apply a bone colormap (CT-like) to the image"""
        return cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    
    def resize_image(self, img, target_size=(512, 512)):
        """Resize image to target size"""
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    def update_slice(self, *args):
        """Update the displayed slice"""
        if self.image_data is None:
            return
        
        try:
            self.indice_corte = int(self.slice_slider.get())
            
            # Get the correct slice based on orientation
            if self.corte_actual == "Axial":
                slice_data = self.image_data[:, :, self.indice_corte]
                overlay_slice = self.overlay_data[:, :, self.indice_corte]
                max_slice = self.depth - 1
            elif self.corte_actual == "Sagittal":
                slice_data = self.image_data[self.indice_corte, :, :]
                overlay_slice = self.overlay_data[self.indice_corte, :, :]
                max_slice = self.width - 1
            else:  # Coronal
                slice_data = self.image_data[:, self.indice_corte, :]
                overlay_slice = self.overlay_data[:, self.indice_corte, :]
                max_slice = self.height - 1
            
            # Update slice label
            self.slice_label.config(text=f"Slice: {self.indice_corte}/{max_slice}")
            
            # Process the image
            normalized = self.normalize_image(slice_data)
            colormap = self.apply_colormap(normalized)
            
            # Blend with overlay
            overlay_normalized = (overlay_slice > 0).astype(np.uint8) * 255
            overlay_rgb = np.zeros((*overlay_normalized.shape, 3), dtype=np.uint8)
            
            # Draw the points with their colors
            for point in self.draw_points:
                x, y, z = point['x'], point['y'], point['z']
                color = point.get('color', self.draw_color)  # Get the color for the point
                if self.corte_actual == "Axial":
                    overlay_rgb[y, x] = color  # Axial view uses x, y
                elif self.corte_actual == "Sagittal":
                    overlay_rgb[y, z] = color  # Sagittal view uses y, z
                else:  # Coronal
                    overlay_rgb[x, z] = color  # Coronal view uses x, z
            
            # Resize both
            resized = self.resize_image(colormap)
            overlay_resized = self.resize_image(overlay_rgb)
            
            # Blend images
            alpha = 0.7
            mask = (overlay_resized > 0).any(axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            combined = np.where(mask_3d, cv2.addWeighted(resized, 1-alpha, overlay_resized, alpha, 0), resized)
            
            # Save current display image for drawing
            self.current_display_img = combined.copy()
            
            # Convert to PIL Image and display
            img = Image.fromarray(combined)
            #img = img.rotate(90, expand=True)
            img_tk = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.config(width=img.width, height=img.height)
            if hasattr(self, 'img_on_canvas'):
                self.canvas.delete(self.img_on_canvas)
            self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference
            
        except Exception as e:
            self.status_var.set(f"Error updating slice: {str(e)}")
    
    def choose_color(self):
        """Open a color chooser dialog to select drawing color"""
        color = colorchooser.askcolor(title="Choose Drawing Color", initialcolor=self.draw_color)
        if color[1]:  # color is a tuple ((r,g,b), hexstring)
            self.draw_color = tuple(int(c) for c in color[0])
            self.color_preview.config(bg=color[1])
            self.status_var.set(f"Draw color set to {color[1]}")
    
    def set_brush_size(self, size):
        """Set the drawing brush size"""
        self.draw_radius = size
        self.status_var.set(f"Brush size set to {size}px")
    
    def start_draw(self, event):
        """Start drawing on mouse press"""
        if not self.draw_mode_var.get() or self.image_data is None:
            return
        
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
        # Draw a single point
        self.draw(event)
    
    def draw(self, event):
        """Draw on the image as mouse moves"""
        if not self.drawing or self.current_display_img is None:
            return
    
        # Get current mouse position in canvas coordinates
        canvas_x, canvas_y = event.x, event.y
    
        # Draw line between last position and current position on display image
        cv2.line(self.current_display_img, (self.last_x, self.last_y), (canvas_x, canvas_y), 
                self.draw_color, self.draw_radius * 2)
    
        # Update display
        img = Image.fromarray(self.current_display_img)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.img_on_canvas, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference
    
        # Get the dimensions of the current slice
        if self.corte_actual == "Axial":
            slice_width, slice_height = self.width, self.height
        elif self.corte_actual == "Sagittal":
            slice_width, slice_height = self.depth, self.height
        else:  # Coronal
            slice_width, slice_height = self.width, self.depth
    
        # Convert canvas coordinates to slice coordinates
        display_width, display_height = 512, 512  # Your resize dimensions
        slice_x = int(canvas_x * slice_width / display_width)
        slice_y = int(canvas_y * slice_height / display_height)
    
        # Map to 3D coordinates based on current view
        if self.corte_actual == "Axial":
            x_3d, y_3d, z_3d = slice_x, slice_y, self.indice_corte
        elif self.corte_actual == "Sagittal":
            x_3d, y_3d, z_3d = self.indice_corte, slice_y, slice_x
        else:  # Coronal
            x_3d, y_3d, z_3d = slice_x, self.indice_corte, slice_y
    
        # Ensure coordinates are within bounds
        x_3d = max(0, min(x_3d, self.width - 1))
        y_3d = max(0, min(y_3d, self.height - 1))
        z_3d = max(0, min(z_3d, self.depth - 1))
    
        # Update overlay data
        self.overlay_data[x_3d, y_3d, z_3d] = 1
    
        # Store drawn point
        self.draw_points.append({
            'x': int(x_3d),
            'y': int(y_3d),
            'z': int(z_3d),
            'color': self.draw_color
        })
    
        # Update coordinate display
        self.coord_var.set(f"Drawn at: x={x_3d}, y={y_3d}, z={z_3d} (View: {self.corte_actual})")
    
        # Remember the last position
        self.last_x, self.last_y = canvas_x, canvas_y

    def stop_draw(self, event):
        """Stop drawing on mouse release"""
        self.drawing = False
    
    def clear_drawings(self):
        """Clear all drawings"""
        if self.image_data is None:
            return
            
        if messagebox.askyesno("Clear Drawings", "Are you sure you want to clear all drawings?"):
            self.overlay_data = np.zeros_like(self.image_data)
            self.draw_points = []
            self.update_slice()
            self.status_var.set("Drawings cleared")
    
    def save_drawings(self):
        """Save drawing points to a JSON file"""
        if not self.draw_points:
            messagebox.showinfo("No Drawings", "There are no drawings to save.")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Save Drawing Points"
            )
            
            if not file_path:
                return
                
            with open(file_path, 'w') as f:
                json.dump({
                    'original_image': os.path.basename(self.file_path),
                    'dimensions': [self.width, self.height, self.depth],
                    'points': self.draw_points
                }, f, indent=2)
                
            self.status_var.set(f"Drawings saved to {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save drawings: {str(e)}")
    
    def load_drawings(self):
        """Load drawing points from a JSON file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON Files", "*.json")],
                title="Load Drawing Points"
            )
            
            if not file_path:
                return
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Verify dimensions match
            if data['dimensions'] != [self.width, self.height, self.depth]:
                messagebox.showwarning(
                    "Dimension Mismatch",
                    "The dimensions of the saved drawings do not match the current image."
                )
                return
                
            # Load points
            self.draw_points = data['points']
            
            # Recreate overlay data
            self.overlay_data = np.zeros_like(self.image_data)
            self.overlay_colors = {}

            for point in self.draw_points:
                x, y, z = point['x'], point['y'], point['z']
                color = point.get('color', self.draw_color)
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
                    self.overlay_data[x, y, z] = 1
                    self.overlay_colors[(x,y,z)] = color
                    
            # Update display
            self.update_slice()
            self.status_var.set(f"Drawings loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load drawings: {str(e)}")

    def show_about(self):
        """Show the about dialog"""
        messagebox.showinfo(
            "About",
            "NIfTI Viewer with 3D Visualization and Drawing\n\n"
            "A simple viewer for NIfTI format neuroimaging data.\n"
            "Features:\n"
            "- 2D slice viewing (Axial, Sagittal, Coronal)\n"
            "- Drawing tools to mark regions of interest\n"
            "- Tracking of 3D coordinates for markers\n"
            "- Save/load drawing coordinates\n"
            "- CT-like colormap visualization"
        )
    
    def visualize_3d(self):
        """Create a 3D visualization of the NIfTI data using VTK"""
        if self.image_data is None:
            messagebox.showinfo("No Data", "Please load a NIfTI image first.")
            return
    
        try:
            self.status_var.set("Creating 3D visualization...")
            self.root.update_idletasks()
        
            # Create a copy of the data for processing
            volume_data = self.image_data.copy()
        
            # Normalize data to 0-255 range
            volume_min = np.min(volume_data)
            volume_max = np.max(volume_data)
            volume_data = ((volume_data - volume_min) / (volume_max - volume_min) * 255).astype(np.uint8)
        
            # Create a VTK image data
            volume = vtk.vtkImageData()
            volume.SetDimensions(self.width, self.height, self.depth)
            volume.SetSpacing(1.0, 1.0, 1.0)
            volume.SetOrigin(0.0, 0.0, 0.0)
            volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
            # Fill the VTK image with data
            vtk_data = numpy_to_vtk(volume_data.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Add overlay data if available (drawn regions)
            if np.any(self.overlay_data > 0):
                # Create a mask of the drawn regions
                overlay_mask = (self.overlay_data > 0).astype(np.uint8) * 255
                # Dilate slightly to make it more visible in 3D
                kernel = np.ones((3, 3, 3), np.uint8)
                overlay_mask = np.array([cv2.dilate(overlay_mask[:, :, i], kernel[:, :, 1], iterations=1) 
                                        for i in range(overlay_mask.shape[2])]).transpose(1, 2, 0)
            
                # Blend overlay with volume data
                r, g, b = self.draw_color
                color_factor = 0.8  # Strength of coloring
                for i in range(overlay_mask.shape[0]):
                    for j in range(overlay_mask.shape[1]):
                        for k in range(overlay_mask.shape[2]):
                            if overlay_mask[i, j, k] > 0:
                                # Blend color with original intensity
                                orig_val = volume_data[i, j, k]
                                volume_data[i, j, k] = int(orig_val * (1 - color_factor) + 
                                                        (r + g + b) / 3 * color_factor)
            
                # Update volume data with overlay
                vtk_data = numpy_to_vtk(volume_data.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Create a volume mapper and property
            volume_mapper = vtk.vtkSmartVolumeMapper()
            volume_mapper.SetInputData(volume)
        
            volume_property = vtk.vtkVolumeProperty()
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
        
            # Create and set transfer functions for CT-like appearance
            color_function = vtk.vtkColorTransferFunction()
            opacity_function = vtk.vtkPiecewiseFunction()
        
            # Setup color transfer function (CT-like grayscale)
            color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)      # Black for air/background
            color_function.AddRGBPoint(50, 0.3, 0.3, 0.3)     # Dark gray for soft tissue
            color_function.AddRGBPoint(150, 0.8, 0.8, 0.8)    # Light gray for bone
            color_function.AddRGBPoint(255, 1.0, 1.0, 1.0)    # White for dense bone
        
            # Add color for overlay (if any drawn regions)
            if np.any(self.overlay_data > 0):
                # Add a color hint for the drawn regions
                r, g, b = self.draw_color
                color_function.AddRGBPoint(200, r/255, g/255, b/255)
        
            # Setup opacity transfer function
            opacity_function.AddPoint(0, 0.0)     # Fully transparent for background
            opacity_function.AddPoint(40, 0.0)    # Still transparent for air
            opacity_function.AddPoint(80, 0.2)    # Slightly visible for soft tissue
            opacity_function.AddPoint(150, 0.4)   # More opaque for bone
            opacity_function.AddPoint(255, 0.8)   # Most opaque for dense bone
        
            # If there are drawn regions, make them more visible
            if np.any(self.overlay_data > 0):
                opacity_function.AddPoint(200, 0.9)  # Make drawn regions very visible
        
            # Set the color and opacity functions
            volume_property.SetColor(color_function)
            volume_property.SetScalarOpacity(opacity_function)
        
            # Set the gradient opacity for edge enhancement
            gradient_opacity = vtk.vtkPiecewiseFunction()
            gradient_opacity.AddPoint(0, 0.0)
            gradient_opacity.AddPoint(90, 0.5)
            gradient_opacity.AddPoint(255, 1.0)
            volume_property.SetGradientOpacity(gradient_opacity)
        
            # Create the volume
            actor_volume = vtk.vtkVolume()
            actor_volume.SetMapper(volume_mapper)
            actor_volume.SetProperty(volume_property)
        
            # Create a rendering window and renderer
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
        
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetSize(800, 600)
            render_window.SetWindowName(f"3D Visualization: {os.path.basename(self.file_path)}")
        
            # Add the volume to the renderer
            renderer.AddVolume(actor_volume)
        
            # Set up camera for a good initial view
            camera = renderer.GetActiveCamera()
            camera.SetPosition(0, -400, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            renderer.ResetCamera()
        
            # Create interaction
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
        
            # Add orientation marker (axes)
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(50, 50, 50)
            axes.SetXAxisLabelText("X")
            axes.SetYAxisLabelText("Y")
            axes.SetZAxisLabelText("Z")
            axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
            axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
            axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
            axes_widget = vtk.vtkOrientationMarkerWidget()
            axes_widget.SetOrientationMarker(axes)
            axes_widget.SetInteractor(interactor)
            axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
            axes_widget.SetEnabled(1)
            axes_widget.InteractiveOff()
        
            # Setup interaction style
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
        
            # Add a text display for information
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"File: {os.path.basename(self.file_path)}\n"
                                f"Dimensions: {self.width}x{self.height}x{self.depth}\n"
                                f"Use mouse to rotate, Ctrl+mouse to pan, Scroll to zoom")
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            text_actor.SetPosition(10, 10)
            renderer.AddActor2D(text_actor)
        
            # Initialize and start the interactor
            interactor.Initialize()
            render_window.Render()
        
            self.status_var.set("3D visualization ready")
            interactor.Start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create 3D visualization: {str(e)}")
            self.status_var.set("Error creating 3D visualization")

#Segmentación y preprocesamiento 

    def show_segmentation_options(self, algorithm):
        """Muestra opciones de configuración para el algoritmo de segmentación seleccionado"""
        self.seg_window = tk.Toplevel(self.root)
        self.seg_window.title(f"Opciones de {algorithm}")
        self.seg_window.geometry("400x300")
        self.seg_window.transient(self.root)
        self.seg_window.grab_set()
    
        frame = ttk.Frame(self.seg_window, padding="10")
        frame.pack(fill="both", expand=True)
    
        # Opciones específicas para cada algoritmo
        if algorithm == "Umbralización":
            ttk.Label(frame, text="Umbral mínimo:").grid(row=0, column=0, sticky="w", pady=5)
            self.thresh_min_var = tk.DoubleVar(value=0.3)
            ttk.Scale(frame, from_=0, to=1, variable=self.thresh_min_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=tk.StringVar(value=lambda: f"{self.thresh_min_var.get():.2f}")).grid(row=0, column=2, padx=5)
        
            ttk.Label(frame, text="Umbral máximo:").grid(row=1, column=0, sticky="w", pady=5)
            self.thresh_max_var = tk.DoubleVar(value=0.7)
            ttk.Scale(frame, from_=0, to=1, variable=self.thresh_max_var, 
                    orient="horizontal").grid(row=1, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=tk.StringVar(value=lambda: f"{self.thresh_max_var.get():.2f}")).grid(row=1, column=2, padx=5)
        
        elif algorithm == "Crecimiento":
            ttk.Label(frame, text="Tolerancia:").grid(row=0, column=0, sticky="w", pady=5)
            self.tolerance_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0.01, to=0.5, variable=self.tolerance_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=tk.StringVar(value=lambda: f"{self.tolerance_var.get():.2f}")).grid(row=0, column=2, padx=5)
        
            ttk.Label(frame, text="Para seleccionar un punto semilla:").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
            ttk.Label(frame, text="1. Haga clic en 'Seleccionar semilla'").grid(row=2, column=0, columnspan=3, sticky="w")
            ttk.Label(frame, text="2. Luego haga clic sobre la imagen").grid(row=3, column=0, columnspan=3, sticky="w")
        
            self.seed_point = None
            ttk.Button(frame, text="Seleccionar semilla", 
                    command=self.enable_seed_selection).grid(row=4, column=0, columnspan=3, pady=10)
        
        elif algorithm == "K-Means":
            ttk.Label(frame, text="Número de clusters (K):").grid(row=0, column=0, sticky="w", pady=5)
            self.k_var = tk.IntVar(value=3)
            ttk.Spinbox(frame, from_=2, to=10, textvariable=self.k_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Máximo de iteraciones:").grid(row=1, column=0, sticky="w", pady=5)
            self.max_iter_var = tk.IntVar(value=100)
            ttk.Spinbox(frame, from_=10, to=500, textvariable=self.max_iter_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
    
        # Botones comunes
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
    
        ttk.Button(button_frame, text="Ejecutar", 
                command=lambda: self.run_segmentation(algorithm)).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancelar", 
                command=self.seg_window.destroy).pack(side="left", padx=10)
    
        # Ajustar el grid
        frame.columnconfigure(1, weight=1)

    def enable_seed_selection(self):
        """Habilita la selección de punto semilla para el crecimiento de regiones"""
        self.seg_window.withdraw()  # Oculta la ventana de opciones temporalmente
        self.status_var.set("Haga clic para seleccionar un punto semilla...")
        self.root.update_idletasks()

        self.canvas.unbind("<Button-1>")

        # Reemplaza temporalmente con nuestra función de selección de semilla
        self.canvas.bind("<Button-1>", self.select_seed_point)

        self.canvas.focus_set()

    def select_seed_point(self, event):
        """Captura el punto semilla seleccionado"""
        # Convertir coordenadas del canvas a coordenadas del volumen
        canvas_x, canvas_y = event.x, event.y
    
        # Obtener dimensiones de la visualización actual
        if self.corte_actual == "Axial":
            slice_width, slice_height = self.width, self.height
        elif self.corte_actual == "Sagittal":
            slice_width, slice_height = self.depth, self.height
        else:  # Coronal
            slice_width, slice_height = self.width, self.depth
    
        # Convertir a coordenadas de la imagen
        display_width, display_height = 512, 512  # Dimensiones tras el resize
        slice_x = int(canvas_x * slice_width / display_width)
        slice_y = int(canvas_y * slice_height / display_height)
    
        # Convertir a coordenadas 3D
        if self.corte_actual == "Axial":
            x_3d, y_3d, z_3d = slice_x, slice_y, self.indice_corte
        elif self.corte_actual == "Sagittal":
            x_3d, y_3d, z_3d = self.indice_corte, slice_y, slice_x
        else:  # Coronal
            x_3d, y_3d, z_3d = slice_x, self.indice_corte, slice_y
    
        self.seed_point = (x_3d, y_3d, z_3d)
        # Mostrar marcador
        self.draw_seed_marker(x_3d, y_3d, z_3d)

        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self.start_draw)
    
        # Mostrar la ventana de opciones nuevamente
        self.seg_window.deiconify()
        self.status_var.set(f"Punto semilla seleccionado en: ({x_3d}, {y_3d}, {z_3d})")

        self.root.update_idletasks()
    
    def draw_seed_marker(self, x, y, z):
        """Dibuja un marcador en la posición seleccionada"""
        if self.corte_actual == "Axial":
            display_x = int(x * 512 / self.width)
            display_y = int(y * 512 / self.height)
        elif self.corte_actual == "Sagittal":
            display_x = int(z * 512 / self.depth)
            display_y = int(y * 512 / self.height)
        else:  # Coronal
            display_x = int(x * 512 / self.width)
            display_y = int(z * 512 / self.depth)

        # Dibujar un pequeño círculo rojo
        marker_id = self.canvas.create_oval(
            display_x-5, display_y-5,
            display_x+5, display_y+5,
            outline="red", width=2
        )

        # Eliminar el marcador después de 3 segundos
        self.root.after(3000, lambda: self.canvas.delete(marker_id))

    def run_segmentation(self, algorithm):
        """Ejecuta el algoritmo de segmentación seleccionado"""
        if self.image_data is None:
            messagebox.showerror("Error", "No hay imagen cargada")
            return
    
        try:
            self.status_var.set(f"Ejecutando segmentación con {algorithm}...")
            self.root.update_idletasks()
        
            # Crear una copia de los datos para no modificar los originales
            segmentation_result = np.zeros_like(self.image_data)
        
            # Ejecutar el algoritmo apropiado
            if algorithm == "Umbralización":
                min_val = np.min(self.image_data)
                max_val = np.max(self.image_data)
                min_threshold = min_val + self.thresh_min_var.get() * (max_val - min_val)
                max_threshold = min_val + self.thresh_max_var.get() * (max_val - min_val)
            
                segmentation_result = self.threshold_segmentation(min_threshold, max_threshold)
            
            elif algorithm == "Crecimiento":
                if self.seed_point is None:
                    messagebox.showerror("Error", "Debe seleccionar un punto semilla")
                    return
                
                segmentation_result = self.region_growing(self.seed_point, self.tolerance_var.get())
            
            elif algorithm == "K-Means":
                segmentation_result = self.kmeans_segmentation(self.k_var.get(), self.max_iter_var.get())
        
            # Mostrar resultado
            self.show_segmentation_result(segmentation_result, algorithm)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar la segmentación: {str(e)}")
            self.status_var.set("Error en la segmentación")
        
        # Cerrar ventana de opciones
        self.seg_window.destroy()

    def threshold_segmentation(self, min_threshold, max_threshold):
        """Implementa segmentación por umbralización"""
        result = np.zeros_like(self.image_data)
        mask = (self.image_data >= min_threshold) & (self.image_data <= max_threshold)
        result[mask] = 1
        return result

    def region_growing(self, seed_point, tolerance):
        """Implementa segmentación por crecimiento de regiones desde cero"""
        # Obtener coordenadas del punto semilla
        x, y, z = seed_point
    
        # Obtener el valor del punto semilla
        seed_value = self.image_data[x, y, z]
    
        # Calcular rango de tolerancia
        min_val = np.min(self.image_data)
        max_val = np.max(self.image_data)
        tolerance_range = tolerance * (max_val - min_val)
    
        # Crear máscara para el resultado
        result = np.zeros_like(self.image_data)
    
        # Crear array para controlar los puntos visitados
        processed = np.zeros_like(self.image_data, dtype=bool)
    
        # Lista de puntos a procesar (comienza con la semilla)
        points_queue = [seed_point]
    
        # Dirección de vecinos (6-conectividad: arriba, abajo, izquierda, derecha, adelante, atrás)
        neighbors = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]
    
        # Mientras haya puntos por procesar
        while points_queue:
            current_x, current_y, current_z = points_queue.pop(0)
        
            # Si está fuera de los límites o ya fue procesado, continuar
            if (current_x < 0 or current_x >= self.width or
                current_y < 0 or current_y >= self.height or
                current_z < 0 or current_z >= self.depth or
                processed[current_x, current_y, current_z]):
                continue
        
            # Marcar como procesado
            processed[current_x, current_y, current_z] = True
        
            # Obtener valor del punto actual
            current_value = self.image_data[current_x, current_y, current_z]
        
            # Si está dentro del rango de tolerancia, agregar a la región
            if abs(current_value - seed_value) <= tolerance_range:
                result[current_x, current_y, current_z] = 1
            
                # Agregar vecinos a la cola
                for dx, dy, dz in neighbors:
                    new_x, new_y, new_z = current_x + dx, current_y + dy, current_z + dz
                    if (0 <= new_x < self.width and 
                        0 <= new_y < self.height and 
                        0 <= new_z < self.depth and 
                        not processed[new_x, new_y, new_z]):
                        points_queue.append((new_x, new_y, new_z))
    
        return result

    def kmeans_segmentation(self, k, max_iterations=100):
        """Implementa segmentación por K-Means optimizado usando NumPy"""
        # Aplanar y normalizar datos
        flattened_data = self.image_data.flatten()
        min_val = flattened_data.min()
        max_val = flattened_data.max()
        normalized_data = (flattened_data - min_val) / (max_val - min_val)
    
        # Inicializar centroides aleatoriamente
        np.random.seed(42)
        centroids = np.random.rand(k)

        for _ in range(max_iterations):
            old_centroids = centroids.copy()
        
            # Asignar cada punto al centroide más cercano (vectorizado)
            distances = np.abs(normalized_data[:, np.newaxis] - centroids[np.newaxis, :])
            labels = np.argmin(distances, axis=1)
        
            # Actualizar centroides (vectorizado)
            for j in range(k):
                if np.any(labels == j):
                    centroids[j] = normalized_data[labels == j].mean()
        
            # Criterio de convergencia
            if np.allclose(centroids, old_centroids, atol=1e-4):
                break

        # Reconstruir la imagen segmentada
        result = centroids[labels]  # Asigna el valor del centroide a cada pixel
        result = result.reshape(self.image_data.shape)
    
        # Escalar de nuevo al rango original (opcional, si quieres visualizar)
        result = (result - result.min()) / (result.max() - result.min())  # Normalizar a 0-1
        result = (result * 255).astype(np.uint8)

        return result

    def show_segmentation_result(self, result, algorithm):
        """Muestra el resultado de la segmentación en una nueva ventana"""
        # Crear una nueva ventana
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Resultado de Segmentación: {algorithm}")
        result_window.geometry("800x700")
    
        # Variables para la ventana de resultados
        self.result_data = result
        self.result_slice_type = "Axial"
        self.result_slice_index = self.depth // 2 if self.result_slice_type == "Axial" else (
            self.width // 2 if self.result_slice_type == "Sagittal" else self.height // 2)
    
        # Frame para controles
        control_frame = ttk.Frame(result_window)
        control_frame.pack(fill="x", padx=10, pady=5)
    
        # Botones para cambiar vista
        ttk.Button(control_frame, text="Axial", 
                command=lambda: self.change_result_view("Axial", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Sagittal", 
                command=lambda: self.change_result_view("Sagittal", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Coronal", 
                command=lambda: self.change_result_view("Coronal", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="3D View", 
                command=lambda: self.visualize_segmentation_3d(result)).pack(side="left", padx=5)
    
        # Slider para navegación
        slice_frame = ttk.Frame(result_window)
        slice_frame.pack(fill="x", padx=10, pady=5)
    
        self.result_slider = ttk.Scale(slice_frame, from_=0, to=100, orient="horizontal", 
                                    command=lambda v: self.update_result_slice(v, result_window))
        self.result_slider.pack(fill="x", padx=10, pady=5)
    
        self.result_slice_label = ttk.Label(slice_frame, text="Slice: 0/0")
        self.result_slice_label.pack(pady=2)
    
        # Canvas para mostrar la imagen
        canvas_frame = ttk.Frame(result_window)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        self.result_canvas = tk.Canvas(canvas_frame)
        self.result_canvas.pack(fill="both", expand=True)
    
        # Botones adicionales
        btn_frame = ttk.Frame(result_window)
        btn_frame.pack(fill="x", padx=10, pady=5)
    
        ttk.Button(btn_frame, text="Aplicar como marcado", 
                command=lambda: self.apply_segmentation_as_overlay(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar a NIfTI", 
                command=lambda: self.export_segmentation(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cerrar", 
                command=result_window.destroy).pack(side="right", padx=5)
    
        # Configurar slider según la orientación
        self.setup_result_slider()
    
        # Mostrar la primera imagen
        self.update_result_slice(self.result_slice_index, result_window)

    def change_result_view(self, view_type, window):
        """Cambia la orientación de visualización del resultado"""
        self.result_slice_type = view_type
    
        # Resetear el índice de corte según la orientación
        if view_type == "Axial":
            self.result_slice_index = self.depth // 2
        elif view_type == "Sagittal":
            self.result_slice_index = self.width // 2
        else:  # Coronal
            self.result_slice_index = self.height // 2
    
        # Actualizar slider y vista
        self.setup_result_slider()
        self.update_result_slice(self.result_slice_index, window)

    def setup_result_slider(self):
        """Configura el slider según la orientación actual"""
        if self.result_slice_type == "Axial":
            max_slice = self.depth - 1
        elif self.result_slice_type == "Sagittal":
            max_slice = self.width - 1
        else:  # Coronal
            max_slice = self.height - 1
    
        self.result_slider.config(from_=0, to=max_slice)
        self.result_slider.set(self.result_slice_index)

    def update_result_slice(self, value, window):
        """Actualiza la visualización del corte del resultado"""
        if isinstance(value, str):
            value = float(value)
        self.result_slice_index = int(value)
    
        # Obtener el corte según la orientación
        if self.result_slice_type == "Axial":
            slice_data = self.result_data[:, :, self.result_slice_index]
            original_slice = self.image_data[:, :, self.result_slice_index]
            max_slice = self.depth - 1
        elif self.result_slice_type == "Sagittal":
            slice_data = self.result_data[self.result_slice_index, :, :]
            original_slice = self.image_data[self.result_slice_index, :, :]
            max_slice = self.width - 1
        else:  # Coronal
            slice_data = self.result_data[:, self.result_slice_index, :]
            original_slice = self.image_data[:, self.result_slice_index, :]
            max_slice = self.height - 1
    
        # Actualizar etiqueta
        self.result_slice_label.config(text=f"Slice: {self.result_slice_index}/{max_slice}")
    
        # Procesar imagen original
        norm_original = self.normalize_image(original_slice)
        color_original = self.apply_colormap(norm_original)
    
        # Crear overlay coloreado para la segmentación
        unique_labels = np.unique(slice_data)
        num_labels = len(unique_labels)
    
        # Crear mapa de colores para la segmentación
        colormap = {}
        for i, label in enumerate(unique_labels):
            if label == 0:  # Fondo es transparente
                colormap[label] = (0, 0, 0, 0)
            else:
                # Crear colores distintos para cada etiqueta
                hue = (i-1) / max(1, num_labels-1) * 180  # Valores HSV de 0 a 180 para OpenCV
                sat = 255
                val = 255
                bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0][0]
                colormap[label] = (*bgr, 150)  # BGR + alpha
    
        # Crear imagen de overlay
        overlay = np.zeros((*slice_data.shape, 4), dtype=np.uint8)
        for label, color in colormap.items():
            mask = (slice_data == label)
            for c in range(3):  # RGB channels
                overlay[..., c][mask] = color[c]
            overlay[..., 3][mask] = color[3]  # Alpha channel
    
        # Resize ambas imágenes
        display_size = (512, 512)
        color_original_resized = cv2.resize(color_original, display_size)
        overlay_resized = cv2.resize(overlay, display_size)
    
        # Mezclar original con overlay
        result_img = color_original_resized.copy()
        for y in range(display_size[1]):
            for x in range(display_size[0]):
                if overlay_resized[y, x, 3] > 0:  # Si hay algo en el overlay (alpha > 0)
                    alpha = overlay_resized[y, x, 3] / 255.0
                    for c in range(3):
                        result_img[y, x, c] = int(result_img[y, x, c] * (1 - alpha) + overlay_resized[y, x, c] * alpha)
    
        # Mostrar en el canvas
        img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
    
        self.result_canvas.config(width=display_size[0], height=display_size[1])
        if hasattr(self, 'result_img_on_canvas'):
            self.result_canvas.delete(self.result_img_on_canvas)
        self.result_img_on_canvas = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.result_canvas.image = img_tk  # Mantener referencia

    def apply_segmentation_as_overlay(self, segmentation):
        """Aplica el resultado de la segmentación como una capa de dibujo"""
        if messagebox.askyesno("Aplicar Segmentación", 
                            "¿Desea aplicar la segmentación como marcado en la imagen original?"):
            # Identificar voxels segmentados
            segmented_indices = np.where(segmentation > 0)
        
            # Convertir a puntos de dibujo
            for i in range(len(segmented_indices[0])):
                x, y, z = segmented_indices[0][i], segmented_indices[1][i], segmented_indices[2][i]
            
                # Crear un punto con el color actual
                self.overlay_data[x, y, z] = 1
                self.draw_points.append({
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'color': self.draw_color
                })
        
            # Actualizar visualización
            self.update_slice()
            self.status_var.set("Segmentación aplicada como marcado")

    def export_segmentation(self, segmentation):
        """Exporta el resultado de la segmentación como archivo NIfTI"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".nii.gz",
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Guardar Segmentación como NIfTI"
            )
        
            if not file_path:
                return
            
            # Crear un nuevo objeto NIfTI con los datos de segmentación
            segmentation_nii = nib.Nifti1Image(segmentation.astype(np.int16), self.nii_image.affine)
        
            # Guardar el archivo
            nib.save(segmentation_nii, file_path)
        
            self.status_var.set(f"Segmentación guardada en {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar segmentación: {str(e)}")

    def visualize_segmentation_3d(self, segmentation):
        """Crea una visualización 3D del resultado de la segmentación"""
        try:
            self.status_var.set("Creando visualización 3D de la segmentación...")
        
            # Crear un nuevo volumen para VTK
            volume = vtk.vtkImageData()
            volume.SetDimensions(self.width, self.height, self.depth)
            volume.SetSpacing(1.0, 1.0, 1.0)
            volume.SetOrigin(0.0, 0.0, 0.0)
            volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
            # Convertir datos de segmentación a formato VTK
            segmentation_data = segmentation.flatten().astype(np.uint8)
            vtk_data = numpy_to_vtk(segmentation_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            volume.GetPointData().GetScalars().DeepCopy(vtk_data)
        
            # Crear un mapeador de contorno para mostrar las superficies de segmentación
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(volume)
        
            # Encontrar todos los valores únicos de etiquetas (excluyendo 0 que es el fondo)
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels > 0]
        
            if len(unique_labels) == 0:
                messagebox.showinfo("Sin datos", "No hay regiones segmentadas para visualizar en 3D.")
                return
        
            # Extraer un contorno por cada etiqueta
            for i, label in enumerate(unique_labels):
                contour.SetValue(i, label)
        
            # Crear mapeador de superficie
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOn()
        
            # Crear mapa de colores
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(len(unique_labels) + 1)
            lut.SetTableRange(0, len(unique_labels))
            lut.Build()
        
            # Configurar colores para cada etiqueta
            for i, label in enumerate(unique_labels):
                # Usar HSV para generar colores distintos
                hue = float(i) / len(unique_labels)
                lut.SetTableValue(i, *[*colorsys.hsv_to_rgb(hue, 1.0, 1.0), 1.0])
        
            mapper.SetLookupTable(lut)
        
            # Crear actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
        
            # Configurar propiedades
            actor.GetProperty().SetOpacity(0.7)
            actor.GetProperty().SetSpecular(0.3)
            # Configuración de la escena
            renderer = vtk.vtkRenderer()
            renderer.AddActor(actor)
            renderer.SetBackground(0.2, 0.2, 0.2)  # Fondo gris oscuro
        
            # Configurar la ventana de renderizado
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetSize(800, 600)
            render_window.SetWindowName("Visualización 3D de Segmentación")
        
            # Configurar interactor
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
        
            # Configurar estilo de interacción
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
        
            # Inicializar y ajustar cámara
            renderer.ResetCamera()
            camera = renderer.GetActiveCamera()
            camera.Elevation(30)
            camera.Azimuth(30)
            camera.Zoom(1.2)
        
            # Iniciar visualización
            interactor.Initialize()
            render_window.Render()
        
            # Mostrar mensaje de estado
            self.status_var.set("Visualización 3D generada. Cierre la ventana 3D para continuar.")
        
            # Iniciar el bucle de eventos
            interactor.Start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar la Visualización 3D: {str(e)}")
            self.status_var.set("Error en la visualización")

#Preprocessing
    def show_preprocessing_options(self, filter_type):

        """Muestra opciones de configuración para el filtro seleccionado"""
        self.prep_window = tk.Toplevel(self.root)
        self.prep_window.title(f"Opciones de Filtro {filter_type}")
        self.prep_window.geometry("400x300")
        self.prep_window.transient(self.root)
        self.prep_window.grab_set()
    
        frame = ttk.Frame(self.prep_window, padding="10")
        frame.pack(fill="both", expand=True)
    
        # Opciones específicas para cada filtro
        if filter_type == "Media":
            ttk.Label(frame, text="Tamaño de kernel:").grid(row=0, column=0, sticky="w", pady=5)
            self.kernel_size_var = tk.IntVar(value=3)
            sizes = [3, 5, 7, 9]
            kernel_combobox = ttk.Combobox(frame, textvariable=self.kernel_size_var, values=sizes, state="readonly", width=5)
            kernel_combobox.grid(row=0, column=1, sticky="w", pady=5)
            kernel_combobox.current(0)
        
        elif filter_type == "Mediana":
            ttk.Label(frame, text="Tamaño de kernel:").grid(row=0, column=0, sticky="w", pady=5)
            self.kernel_size_var = tk.IntVar(value=3)
            sizes = [3, 5, 7, 9]
            kernel_combobox = ttk.Combobox(frame, textvariable=self.kernel_size_var, values=sizes, state="readonly", width=5)
            kernel_combobox.grid(row=0, column=1, sticky="w", pady=5)
            kernel_combobox.current(0)
        
        elif filter_type == "Bilateral":
            ttk.Label(frame, text="Tamaño de ventana:").grid(row=0, column=0, sticky="w", pady=5)
            self.window_size_var = tk.IntVar(value=9)
            ttk.Spinbox(frame, from_=3, to=15, textvariable=self.window_size_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Sigma espacial:").grid(row=1, column=0, sticky="w", pady=5)
            self.sigma_space_var = tk.DoubleVar(value=1.5)
            ttk.Spinbox(frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.sigma_space_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Sigma rango:").grid(row=2, column=0, sticky="w", pady=5)
            self.sigma_range_var = tk.DoubleVar(value=50.0)
            ttk.Spinbox(frame, from_=10.0, to=150.0, increment=5.0, textvariable=self.sigma_range_var, width=5).grid(row=2, column=1, sticky="w", pady=5)
        
        elif filter_type == "Anisotropico":
            ttk.Label(frame, text="Iteraciones:").grid(row=0, column=0, sticky="w", pady=5)
            self.iterations_var = tk.IntVar(value=10)
            ttk.Spinbox(frame, from_=1, to=50, textvariable=self.iterations_var, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Kappa (conductancia):").grid(row=1, column=0, sticky="w", pady=5)
            self.kappa_var = tk.DoubleVar(value=50.0)
            ttk.Spinbox(frame, from_=1.0, to=100.0, increment=1.0, textvariable=self.kappa_var, width=5).grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Lambda (paso tiempo):").grid(row=2, column=0, sticky="w", pady=5)
            self.lambda_var = tk.DoubleVar(value=0.25)
            ttk.Spinbox(frame, from_=0.05, to=0.25, increment=0.05, textvariable=self.lambda_var, width=5).grid(row=2, column=1, sticky="w", pady=5)
    
        elif filter_type == "Bordes":
            ttk.Label(frame, text="Umbral bajo:").grid(row=0, column=0, sticky="w", pady=5)
            self.edge_low_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0, to=1, variable=self.edge_low_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
        
            ttk.Label(frame, text="Umbral alto:").grid(row=1, column=0, sticky="w", pady=5)
            self.edge_high_var = tk.DoubleVar(value=0.3)
            ttk.Scale(frame, from_=0, to=1, variable=self.edge_high_var, 
                    orient="horizontal").grid(row=1, column=1, sticky="ew", pady=5)
        
            ttk.Label(frame, text="Tamaño del kernel:").grid(row=2, column=0, sticky="w", pady=5)
            self.edge_kernel_var = tk.IntVar(value=3)
            ttk.Combobox(frame, textvariable=self.edge_kernel_var, 
                        values=[3, 5, 7], width=5).grid(row=2, column=1, sticky="w", pady=5)
        
        elif filter_type == "NLM":
            ttk.Label(frame, text="Tamaño de parche:").grid(row=0, column=0, sticky="w", pady=5)
            self.nlm_patch_size_var = tk.IntVar(value=3)
            ttk.Combobox(frame, textvariable=self.nlm_patch_size_var, 
                        values=[3, 5, 7], width=5).grid(row=0, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Radio de búsqueda:").grid(row=1, column=0, sticky="w", pady=5)
            self.nlm_search_var = tk.IntVar(value=5)
            ttk.Combobox(frame, textvariable=self.nlm_search_var, 
                        values=[5, 7, 9, 11], width=5).grid(row=1, column=1, sticky="w", pady=5)
        
            ttk.Label(frame, text="Parámetro h (fuerza):").grid(row=2, column=0, sticky="w", pady=5)
            self.nlm_h_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0.01, to=0.5, variable=self.nlm_h_var, 
                    orient="horizontal").grid(row=2, column=1, sticky="ew", pady=5)
        
        elif filter_type == "Roberts":
            ttk.Label(frame, text="Umbral para binarización:").grid(row=0, column=0, sticky="w", pady=5)
            self.roberts_threshold_var = tk.DoubleVar(value=0.1)
            ttk.Scale(frame, from_=0.01, to=0.5, variable=self.roberts_threshold_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.roberts_threshold_var).grid(row=0, column=2, padx=5)

        elif filter_type == "LoG":
            ttk.Label(frame, text="Sigma (desviación estándar):").grid(row=0, column=0, sticky="w", pady=5)
            self.log_sigma_var = tk.DoubleVar(value=1.0)
            ttk.Scale(frame, from_=0.5, to=5.0, variable=self.log_sigma_var, 
                    orient="horizontal").grid(row=0, column=1, sticky="ew", pady=5)
            ttk.Label(frame, textvariable=self.log_sigma_var).grid(row=0, column=2, padx=5)
        
            ttk.Label(frame, text="Tamaño de kernel:").grid(row=1, column=0, sticky="w", pady=5)
            self.log_kernel_size_var = tk.IntVar(value=7)
            ttk.Spinbox(frame, from_=3, to=21, increment=2, textvariable=self.log_kernel_size_var, 
                    width=5).grid(row=1, column=1, sticky="w", pady=5)
        

        # Botones comunes
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
    
        ttk.Button(button_frame, text="Aplicar", 
                command=lambda: self.run_preprocessing(filter_type)).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancelar", 
                command=self.prep_window.destroy).pack(side="left", padx=10)
    
        # Ajustar el grid
        frame.columnconfigure(1, weight=1)

    def run_preprocessing(self, filter_type):
        """Ejecuta el filtro de preprocesamiento seleccionado"""
        if self.image_data is None:
            messagebox.showerror("Error", "No hay imagen cargada")
            return
    
        try:
            self.status_var.set(f"Aplicando filtro {filter_type}...")
            self.root.update_idletasks()
        
            # Crear una copia de los datos para no modificar los originales
            preprocessed_data = self.image_data.copy()
        
            # Aplicar el filtro apropiado
            if filter_type == "Media":
                preprocessed_data = self.mean_filter(preprocessed_data, self.kernel_size_var.get())
            elif filter_type == "Mediana":
                preprocessed_data = self.median_filter(preprocessed_data, self.kernel_size_var.get())
            elif filter_type == "Bilateral":
                preprocessed_data = self.bilateral_filter(
                    preprocessed_data, 
                    self.window_size_var.get(),
                    self.sigma_space_var.get(),
                    self.sigma_range_var.get()
                )
            elif filter_type == "Anisotropico":
                preprocessed_data = self.anisotropic_diffusion(
                    preprocessed_data,
                    self.iterations_var.get(),
                    self.kappa_var.get(),
                    self.lambda_var.get()
                )
            
            elif filter_type == "Bordes":
                low_threshold = self.edge_low_var.get()
                high_threshold = self.edge_high_var.get()
                kernel_size = self.edge_kernel_var.get()
                preprocessed_data = self.edge_detection(low_threshold, high_threshold, kernel_size)
        
            elif filter_type == "NLM":
                patch_size = self.nlm_patch_size_var.get()
                search_radius = self.nlm_search_var.get()
                h_param = self.nlm_h_var.get()
                preprocessed_data = self.non_local_means(patch_size, search_radius, h_param)
            
            elif filter_type == "Roberts":
                threshold = self.roberts_threshold_var.get()
                preprocessed_data = self.roberts_edge_detection(preprocessed_data, threshold)
            
            elif filter_type == "LoG":
                sigma = self.log_sigma_var.get()
                kernel_size = self.log_kernel_size_var.get()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                preprocessed_data=self.laplacian_of_gaussian(preprocessed_data, sigma, kernel_size)
            
            # Mostrar resultado
            self.show_preprocessing_result(preprocessed_data, filter_type)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar el filtro: {str(e)}")
            self.status_var.set("Error en el preprocesamiento")
    
        # Cerrar ventana de opciones
        self.prep_window.destroy()

    def mean_filter(self, data, kernel_size):
        """Implementa un filtro de media sin usar bibliotecas externas"""
        # Crear copia de datos
        result = np.zeros_like(data)
    
        # Calcular el desplazamiento desde el centro (radio)
        radius = kernel_size // 2
    
        # Iterar por cada voxel en el volumen
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    # Acumuladores para la media
                    sum_values = 0.0
                    count = 0
                
                    # Iterar por el vecindario del kernel
                    for kz in range(-radius, radius + 1):
                        for ky in range(-radius, radius + 1):
                            for kx in range(-radius, radius + 1):
                                # Coordenadas del vecino
                                nz = z + kz
                                ny = y + ky
                                nx = x + kx
                            
                                # Verificar límites
                                if (0 <= nx < self.width and 
                                    0 <= ny < self.height and 
                                    0 <= nz < self.depth):
                                    # Sumar valor y contar
                                    sum_values += data[nx, ny, nz]
                                    count += 1
                
                    # Calcular media
                    if count > 0:
                        result[x, y, z] = sum_values / count
    
        return result

    def median_filter(self, data, kernel_size):
        """Implementa un filtro de mediana sin usar bibliotecas externas"""
        # Crear copia de datos
        result = np.zeros_like(data)
    
        # Calcular el desplazamiento desde el centro (radio)
        radius = kernel_size // 2
    
        # Iterar por cada voxel en el volumen
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    # Lista para almacenar valores del vecindario
                    neighborhood = []
                
                    # Iterar por el vecindario del kernel
                    for kz in range(-radius, radius + 1):
                        for ky in range(-radius, radius + 1):
                            for kx in range(-radius, radius + 1):
                                # Coordenadas del vecino
                                nz = z + kz
                                ny = y + ky
                                nx = x + kx
                            
                                # Verificar límites
                                if (0 <= nx < self.width and 
                                    0 <= ny < self.height and 
                                    0 <= nz < self.depth):
                                    # Añadir valor a la lista
                                    neighborhood.append(data[nx, ny, nz])
                
                    # Calcular mediana
                    if neighborhood:
                        result[x, y, z] = np.median(neighborhood)
    
        return result

    def bilateral_filter(self, data, window_size, sigma_space, sigma_range):
        """Filtro bilateral corregido para evitar accesos fuera de límites."""
        result = np.zeros_like(data)
        radius = window_size // 2
        kernel_size = 2 * radius + 1  # Tamaño total del kernel

        # Precalcular kernel espacial
        zz, yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        d_squared = xx**2 + yy**2 + zz**2
        spatial_kernel = np.exp(-0.5 * d_squared / (sigma_space ** 2))

        range_gauss_coeff = -0.5 / (sigma_range ** 2)

        for z in range(self.depth):
            self.status_var.set(f"Procesando filtro bilateral: {z+1}/{self.depth}")
            self.root.update_idletasks()

            for y in range(self.height):
                for x in range(self.width):
                    center_value = data[z, y, x]

                    # Calcular límites para la ventana
                    z_min = max(z - radius, 0)
                    z_max = min(z + radius + 1, self.depth)
                    y_min = max(y - radius, 0)
                    y_max = min(y + radius + 1, self.height)
                    x_min = max(x - radius, 0)
                    x_max = min(x + radius + 1, self.width)

                    # Extraer ventana local
                    local_region = data[z_min:z_max, y_min:y_max, x_min:x_max]
                
                    # Crear un kernel espacial personalizado para esta ventana
                    sz, sy, sx = local_region.shape
                
                    # Crear coordenadas para la ventana local
                    temp_z, temp_y, temp_x = np.mgrid[0:sz, 0:sy, 0:sx]
                
                    # Calcular el centro relativo a la ventana local
                    center_z = z - z_min
                    center_y = y - y_min
                    center_x = x - x_min
                
                    # Calcular distancias al cuadrado desde el centro
                    d_sq = (temp_z - center_z)**2 + (temp_y - center_y)**2 + (temp_x - center_x)**2
                
                    # Kernel espacial específico para esta ventana
                    local_spatial = np.exp(-0.5 * d_sq / (sigma_space ** 2))
                
                    # Calcular kernel de rango
                    intensity_diff = local_region - center_value
                    range_kernel = np.exp((intensity_diff ** 2) * range_gauss_coeff)
                
                    # Calcular kernel total
                    total_kernel = local_spatial * range_kernel

                    # Aplicar filtro y normalizar
                    weighted_sum = np.sum(total_kernel * local_region)
                    weight_sum = np.sum(total_kernel)

                    if weight_sum > 0:
                        result[z, y, x] = weighted_sum / weight_sum
                    else:
                        result[z, y, x] = center_value

        return result

    def anisotropic_diffusion(self, data, iterations, kappa, lambda_val):
        """Implementa el filtro de difusión anisotrópica (Perona-Malik)"""
        # Crear copia de datos
        result = data.copy()
    
        # Función de conducción
        def g(gradient, k):
            """Función de conducción de Perona-Malik (preserva bordes de alto contraste)"""
            return np.exp(-(gradient/k)**2)
    
        # Iterar por el número especificado de iteraciones
        for i in range(iterations):
            self.status_var.set(f"Iteración de difusión anisotrópica: {i+1}/{iterations}")
            self.root.update_idletasks()
        
            # Crear una copia temporal para la actualización
            updated = result.copy()
        
            # Iterar por cada voxel en el volumen (excepto bordes)
            for z in range(1, self.depth-1):
                for y in range(1, self.height-1):
                    for x in range(1, self.width-1):
                        # Calcular gradientes en las 6 direcciones
                        nabla_n = result[x, y-1, z] - result[x, y, z]
                        nabla_s = result[x, y+1, z] - result[x, y, z]
                        nabla_e = result[x+1, y, z] - result[x, y, z]
                        nabla_w = result[x-1, y, z] - result[x, y, z]
                        nabla_t = result[x, y, z+1] - result[x, y, z]
                        nabla_b = result[x, y, z-1] - result[x, y, z]
                    
                        # Calcular coeficientes de difusión
                        cn = g(nabla_n, kappa)
                        cs = g(nabla_s, kappa)
                        ce = g(nabla_e, kappa)
                        cw = g(nabla_w, kappa)
                        ct = g(nabla_t, kappa)
                        cb = g(nabla_b, kappa)
                    
                        # Actualizar valor actual según ecuación de difusión
                        updated[x, y, z] = result[x, y, z] + lambda_val * (
                            cn * nabla_n + cs * nabla_s + 
                            ce * nabla_e + cw * nabla_w +
                            ct * nabla_t + cb * nabla_b
                        )
        
            # Actualizar resultado para la siguiente iteración
            result = updated
    
        return result

    def edge_detection(self, low_threshold, high_threshold, kernel_size):
        """Implementa detección de bordes tipo Canny desde cero"""
        # Crear un resultado 3D
        result = np.zeros_like(self.image_data)
    
        # Procesar cada slice
        for z in range(self.depth):
            # Obtener slice
            slice_data = self.image_data[:, :, z]
        
            # Normalizar a rango [0-1]
            slice_norm = self.normalize_0_1(slice_data)
        
            # 1. Suavizado Gaussiano
            smoothed = self.gaussian_blur(slice_norm, kernel_size)
        
            # 2. Cálculo de gradientes
            gx, gy = self.sobel_gradients(smoothed)
        
            # 3. Magnitud del gradiente
            magnitude = np.sqrt(gx**2 + gy**2)
        
            # 4. Dirección del gradiente
            direction = np.arctan2(gy, gx)
        
            # 5. Supresión de no máximos
            suppressed = self.non_maximum_suppression(magnitude, direction)
        
            # 6. Umbralización con histéresis
            min_val = magnitude.min()
            max_val = magnitude.max()
            low = min_val + low_threshold * (max_val - min_val)
            high = min_val + high_threshold * (max_val - min_val)
        
            edges = self.hysteresis_threshold(suppressed, low, high)
        
            # Asignar resultado
            result[:, :, z] = edges
    
        return result
    
    def non_local_means(self, patch_size, search_radius, h_param):
        """Implementa Non-Local Means para reducción de ruido desde cero"""
        # Crear un resultado 3D
        result = np.zeros_like(self.image_data)
    
        # Procesar cada slice (para ahorrar tiempo y memoria)
        for z in range(self.depth):
            # Obtener slice
            slice_data = self.image_data[:, :, z]
        
            # Normalizar a rango [0-1]
            slice_norm = self.normalize_0_1(slice_data)
        
            # Implementación NLM 2D
            denoised = self.nlm_2d(slice_norm, patch_size, search_radius, h_param)
        
            # Asignar resultado
            result[:, :, z] = denoised * (slice_data.max() - slice_data.min()) + slice_data.min()
    
        return result

    def normalize_0_1(self, data):
        """Normaliza datos al rango [0-1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def gaussian_blur(self, image, kernel_size):
        """Implementa desenfoque gaussiano"""
        # Crear kernel gaussiano
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel_1d = np.array([np.exp(-(x - kernel_size//2)**2/(2*sigma**2)) for x in range(kernel_size)])
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalizar
    
        # Aplicar convolución separable (horizontalmente y luego verticalmente)
        temp = np.zeros_like(image)
        result = np.zeros_like(image)
    
        # Convolución horizontal
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = 0
                for k in range(kernel_size):
                    pos = j - kernel_size//2 + k
                    if 0 <= pos < image.shape[1]:
                        val += image[i, pos] * kernel_1d[k]
                temp[i, j] = val
    
        # Convolución vertical
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = 0
                for k in range(kernel_size):
                    pos = i - kernel_size//2 + k
                    if 0 <= pos < image.shape[0]:
                        val += temp[pos, j] * kernel_1d[k]
                result[i, j] = val
    
        return result

    def sobel_gradients(self, image):
        """Calcula gradientes usando operadores Sobel"""
        # Kernels de Sobel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
        # Calcular gradientes
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
    
        # Aplicar convolución
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                gx[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
                gy[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)
    
        return gx, gy

    def non_maximum_suppression(self, magnitude, direction):
        """Suprime valores no máximos en la dirección del gradiente"""
        result = np.zeros_like(magnitude)
        height, width = magnitude.shape
    
        # Convertir ángulos a grados y ajustar a 0-180
        direction = np.degrees(direction) % 180
    
        for i in range(1, height-1):
            for j in range(1, width-1):
                # Obtener ángulo
                angle = direction[i, j]
            
                # Determinar vecinos en la dirección del gradiente
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    # 0 grados - horizontal
                    neighbor1 = magnitude[i, j-1]
                    neighbor2 = magnitude[i, j+1]
                elif 22.5 <= angle < 67.5:
                    # 45 grados
                    neighbor1 = magnitude[i+1, j-1]
                    neighbor2 = magnitude[i-1, j+1]
                elif 67.5 <= angle < 112.5:
                    # 90 grados - vertical
                    neighbor1 = magnitude[i-1, j]
                    neighbor2 = magnitude[i+1, j]
                else:
                    # 135 grados
                    neighbor1 = magnitude[i-1, j-1]
                    neighbor2 = magnitude[i+1, j+1]
            
                # Comprobar si el píxel es máximo en la dirección del gradiente
                if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                    result[i, j] = magnitude[i, j]
    
        return result

    def hysteresis_threshold(self, image, low, high):
        """Umbralización con histéresis para detección de bordes"""
        # Crear máscara de bordes fuertes y débiles
        strong_edges = image >= high
        weak_edges = (image >= low) & (image < high)
    
        # Resultado final
        result = np.zeros_like(image)
        result[strong_edges] = 1
    
        # Lista de píxeles fuertes (semillas)
        height, width = image.shape
        strong_i, strong_j = np.where(strong_edges)
    
        # Conectar bordes débiles a fuertes
        while len(strong_i) > 0:
            # Sacar un píxel fuerte
            i, j = strong_i[0], strong_j[0]
            strong_i = strong_i[1:]
            strong_j = strong_j[1:]
        
            # Verificar los 8 vecinos
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                
                    ni, nj = i + di, j + dj
                    if (0 <= ni < height and 0 <= nj < width and 
                        weak_edges[ni, nj] and result[ni, nj] == 0):
                        # Convertir borde débil a fuerte
                        result[ni, nj] = 1
                        # Añadir a la lista para procesar sus vecinos
                        strong_i = np.append(strong_i, ni)
                        strong_j = np.append(strong_j, nj)
                        # Marcar como procesado
                        weak_edges[ni, nj] = False
    
        return result

    def nlm_2d(self, image, patch_size, search_radius, h_param):
        """Implementación simplificada de Non-Local Means para una imagen 2D"""
        height, width = image.shape
        result = np.zeros_like(image)
        h_squared = h_param ** 2
    
        # Calcular la mitad del tamaño del parche
        patch_half = patch_size // 2
    
        # Procesar cada píxel
        for i in range(patch_half, height - patch_half):
            # Mostrar progreso
            if i % 10 == 0:
                self.status_var.set(f"Procesando NLM: {i}/{height} filas...")
                self.root.update_idletasks()
        
            for j in range(patch_half, width - patch_half):
                # Parche central
                patch_center = image[i-patch_half:i+patch_half+1, j-patch_half:j+patch_half+1]
            
                # Inicializar variables
                weighted_sum = 0
                weight_sum = 0
            
                # Limitar área de búsqueda para optimización
                search_start_i = max(patch_half, i - search_radius)
                search_end_i = min(height - patch_half, i + search_radius + 1)
                search_start_j = max(patch_half, j - search_radius)
                search_end_j = min(width - patch_half, j + search_radius + 1)
            
                # Buscar parches similares
                for si in range(search_start_i, search_end_i):
                    for sj in range(search_start_j, search_end_j):
                        # Parche de búsqueda
                        patch_search = image[si-patch_half:si+patch_half+1, sj-patch_half:sj+patch_half+1]
                    
                        # Calcular distancia (diferencia cuadrada)
                        distance = np.sum((patch_center - patch_search) ** 2)
                    
                        # Calcular peso
                        weight = np.exp(-distance / h_squared)
                    
                        # Acumular suma ponderada
                        weighted_sum += weight * image[si, sj]
                        weight_sum += weight
            
                # Calcular valor final (normalizado por suma de pesos)
                result[i, j] = weighted_sum / weight_sum
    
        # Copiar bordes de la imagen original
        for i in range(height):
            for j in range(width):
                if (i < patch_half or i >= height - patch_half or 
                    j < patch_half or j >= width - patch_half):
                    result[i, j] = image[i, j]
    
        return result

    def roberts_edge_detection(self, image_data, threshold):
    
        # Crear una copia para no modificar la imagen original
        result = np.zeros_like(image_data, dtype=np.float32)
    
        # Definir los kernels del operador de Roberts
        roberts_cross_v = np.array([[1, 0], 
                                    [0, -1]])
    
        roberts_cross_h = np.array([[0, 1], 
                                    [-1, 0]])
    
        # Procesar cada slice de la imagen 3D
        for i in range(image_data.shape[0]):
            # Obtener el slice actual
            slice_data = image_data[i, :, :].astype(np.float32)
        
         # Aplicar el operador de Roberts
            vertical = np.zeros_like(slice_data)
            horizontal = np.zeros_like(slice_data)
        
            # Aplicar convolución manual
            rows, cols = slice_data.shape
            for r in range(rows-1):
                for c in range(cols-1):
                    # Gradiente horizontal
                    horizontal[r, c] = np.sum(slice_data[r:r+2, c:c+2] * roberts_cross_h)
                    # Gradiente vertical
                    vertical[r, c] = np.sum(slice_data[r:r+2, c:c+2] * roberts_cross_v)
        
            # Calcular la magnitud del gradiente
            gradient_magnitude = np.sqrt(np.square(horizontal) + np.square(vertical))
        
            # Normalizar a [0, 1]
            if gradient_magnitude.max() > 0:
                gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
            # Aplicar umbral
            gradient_magnitude = np.where(gradient_magnitude > threshold, 1.0, 0.0)
        
            # Guardar resultado
            result[i, :, :] = gradient_magnitude
    
        return result

    def laplacian_of_gaussian(self, preprocessed_data, sigma, kernel_size):

        # Crear kernels
        gaussian = self.gaussian_kernel(kernel_size, sigma)
        laplacian = self.laplacian_kernel(kernel_size)

        # Crear volumen de salida con mismo shape
        output = np.zeros_like(preprocessed_data, dtype=np.uint8)

        # Asumimos que la imagen es 3D (H, W, D)
        for z in range(preprocessed_data.shape[2]):
            slice_2d = preprocessed_data[:, :, z]

            # Suavizado
            smoothed = self.convolution2d(slice_2d, gaussian)

            # Laplaciano
            result = self.convolution2d(smoothed, laplacian)

            # Normalizar
            min_val = np.min(result)
            max_val = np.max(result)
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)
            else:
                result = np.zeros_like(result)

            # Detección de cruces por cero
            zero_crossings = np.zeros_like(result, dtype=np.uint8)
            h, w = result.shape

            for i in range(1, h-1):
                for j in range(1, w-1):
                    neighbors = [
                        result[i-1, j-1], result[i-1, j], result[i-1, j+1],
                        result[i, j-1],                 result[i, j+1],
                        result[i+1, j-1], result[i+1, j], result[i+1, j+1]
                    ]
                    if any(n * result[i, j] < 0 for n in neighbors):
                        zero_crossings[i, j] = 1

            output[:, :, z] = zero_crossings

        return output

    def gaussian_kernel(self, size, sigma):
        """Crea un kernel gaussiano 2D"""
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    def laplacian_kernel(self, size):
        """Crea un kernel laplaciano"""
        laplacian = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        if size > 3:
            pad_size = (size - 3) // 2
            laplacian = np.pad(laplacian, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
        return laplacian

    def convolution2d(self, image, kernel):
        """Convolución 2D desde cero"""
        h_image, w_image = image.shape
        h_kernel, w_kernel = kernel.shape

        h_offset = h_kernel // 2
        w_offset = w_kernel // 2

        padded_image = np.zeros((h_image + 2 * h_offset, w_image + 2 * w_offset))
        padded_image[h_offset:h_offset + h_image, w_offset:w_offset + w_image] = image

        result = np.zeros_like(image, dtype=np.float32)

        for i in range(h_image):
            for j in range(w_image):
                region = padded_image[i:i + h_kernel, j:j + w_kernel]
                result[i, j] = np.sum(region * kernel)
    
        return result
    
    def update_result_processing_slice(self, value, window):
        """Actualiza la visualización del corte del resultado con comparación opcional"""
        if isinstance(value, str):
            value = float(value)
        self.result_slice_index = int(value)
    
        # Obtener el corte según la orientación
        if self.result_slice_type == "Axial":
            filtered_slice = self.result_data[:, :, self.result_slice_index]
            max_slice = self.depth - 1
        elif self.result_slice_type == "Sagittal":
            filtered_slice = self.result_data[self.result_slice_index, :, :]
            max_slice = self.width - 1
        else:  # Coronal
            filtered_slice = self.result_data[:, self.result_slice_index, :]
            max_slice = self.height - 1
    
        # Actualizar etiqueta
        self.result_slice_label.config(text=f"Slice: {self.result_slice_index}/{max_slice}")
    
        norm_filtered = self.normalize_image(filtered_slice)
        color_filtered = self.apply_colormap(norm_filtered)
    
        # Resize ambas imágenes
        display_size = (512, 512)
        color_filtered_resized = cv2.resize(color_filtered, display_size)
    
   
        # Mostrar solo imagen filtrada
        combined_img = color_filtered_resized
        self.result_canvas.config(width=display_size[0], height=display_size[1])
    
        # Mostrar en el canvas
        img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
    
        if hasattr(self, 'result_img_on_canvas'):
            self.result_canvas.delete(self.result_img_on_canvas)
        self.result_img_on_canvas = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.result_canvas.image = img_tk  # Mantener referencia
    
    def show_preprocessing_result(self, result, filter_type):
        """Muestra el resultado del preprocesamiento en una nueva ventana"""
        # Crear una nueva ventana
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Resultado de Preprocesamiento: {filter_type}")
        result_window.geometry("800x700")
    
        # Variables para la ventana de resultados
        self.result_data = result
        self.result_slice_type = "Axial"
        self.result_slice_index = self.depth // 2
    
        # Frame para controles
        control_frame = ttk.Frame(result_window)
        control_frame.pack(fill="x", padx=10, pady=5)
    
        # Botones para cambiar vista
        ttk.Button(control_frame, text="Axial", 
                command=lambda: self.change_result_view("Axial", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Sagittal", 
                command=lambda: self.change_result_view("Sagittal", result_window)).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Coronal", 
                command=lambda: self.change_result_view("Coronal", result_window)).pack(side="left", padx=5)
    
        # Comparación lado a lado
        self.show_comparison_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Mostrar comparación", 
                    variable=self.show_comparison_var, 
                    command=lambda: self.update_result_processing_slice(self.result_slice_index, result_window)).pack(side="left", padx=10)
    
        # Slider para navegación
        slice_frame = ttk.Frame(result_window)
        slice_frame.pack(fill="x", padx=10, pady=5)
    
        self.result_slider = ttk.Scale(slice_frame, from_=0, to=100, orient="horizontal", 
                                  command=lambda v: self.update_result_processing_slice(v, result_window))
        self.result_slider.pack(fill="x", padx=10, pady=5)
    
        self.result_slice_label = ttk.Label(slice_frame, text="Slice: 0/0")
        self.result_slice_label.pack(pady=2)
    
        # Canvas para mostrar la imagen
        canvas_frame = ttk.Frame(result_window)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        self.result_canvas = tk.Canvas(canvas_frame)
        self.result_canvas.pack(fill="both", expand=True)
    
        # Botones adicionales
        btn_frame = ttk.Frame(result_window)
        btn_frame.pack(fill="x", padx=10, pady=5)
    
        ttk.Button(btn_frame, text="Aplicar a imagen", 
                command=lambda: self.apply_preprocessing(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exportar a NIfTI", 
                command=lambda: self.export_preprocessing(result)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cerrar", 
                command=result_window.destroy).pack(side="right", padx=5)
    
        # Configurar slider según la orientación
        self.setup_result_slider()
    
        # Mostrar la primera imagen
        self.update_result_processing_slice(self.result_slice_index, result_window)

    def apply_preprocessing(self, processed_data):
        """Aplica el resultado del preprocesamiento como nueva imagen principal"""
        if messagebox.askyesno("Aplicar Preprocesamiento", 
                        "¿Desea aplicar el resultado como imagen principal?\n" +
                        "Esto reemplazará los datos actuales."):
            # Actualizar datos de la imagen
            self.image_data = processed_data.copy()
        
            # Limpiar dibujos previos
            self.overlay_data = np.zeros_like(self.image_data)
            self.draw_points = []
        
            # Actualizar visualización
            self.update_slice()
            self.status_var.set("Preprocesamiento aplicado como imagen principal")

    def export_preprocessing(self, result):
        """Exporta el resultado del preprocesamiento como archivo NIfTI"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".nii.gz",
                filetypes=[("NIfTI Files", "*.nii *.nii.gz")],
                title="Guardar Resultado como NIfTI"
            )
        
            if not file_path:
                return
        
            # Crear un nuevo objeto NIfTI con los datos procesados
            processed_nii = nib.Nifti1Image(result, self.nii_image.affine)
        
            # Guardar el archivo
            nib.save(processed_nii, file_path)
        
            self.status_var.set(f"Resultado guardado en {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar resultado: {str(e)}")



if __name__ == "__main__":
    root = tk.Tk()
    app = NiftiViewer(root)
    root.mainloop()
