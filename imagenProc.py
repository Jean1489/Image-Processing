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
            overlay_rgb[overlay_normalized > 0] = self.draw_color
            
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
        if not self.drawing:
            return
        
        # Get current mouse position
        x, y = event.x, event.y
        
        # Draw line between last position and current position
        cv2.line(self.current_display_img, (self.last_x, self.last_y), (x, y), self.draw_color, self.draw_radius * 2)
        
        # Create PIL Image from the modified array
        img = Image.fromarray(self.current_display_img)
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the displayed image
        self.canvas.itemconfig(self.img_on_canvas, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference
        
        # Convert display coordinates to original image coordinates
        orig_x = int(x * (self.width if self.corte_actual == "Axial" or self.corte_actual == "Coronal" else self.height) / 512)
        orig_y = int(y * (self.height if self.corte_actual == "Axial" else self.depth) / 512)
        
        # Get 3D coordinates based on view
        if self.corte_actual == "Axial":
            # For axial view, x and y are image coordinates, z is the slice
            x_3d, y_3d, z_3d = orig_x, orig_y, self.indice_corte
            
            # Update overlay data
            if 0 <= x_3d < self.width and 0 <= y_3d < self.height:
                self.overlay_data[x_3d, y_3d, z_3d] = 1
                
        elif self.corte_actual == "Sagittal":
            # For sagittal view, x is the slice, y is vertical, z is horizontal
            x_3d, y_3d, z_3d = self.indice_corte, orig_y, orig_x
            
            # Update overlay data
            if 0 <= y_3d < self.height and 0 <= z_3d < self.depth:
                self.overlay_data[x_3d, y_3d, z_3d] = 1
                
        else:  # Coronal
            # For coronal view, x is horizontal, y is the slice, z is vertical
            x_3d, y_3d, z_3d = orig_x, self.indice_corte, orig_y
            
            # Update overlay data
            if 0 <= x_3d < self.width and 0 <= z_3d < self.depth:
                self.overlay_data[x_3d, y_3d, z_3d] = 1
        
        # Store drawn point with real world coordinates
        self.draw_points.append({
            'x': int(x_3d),
            'y': int(y_3d),
            'z': int(z_3d),
        })
        
        # Update coordinate display
        self.coord_var.set(f"Drawn at: x={x_3d}, y={y_3d}, z={z_3d} (View: {self.corte_actual})")
        
        # Remember the last position
        self.last_x, self.last_y = x, y
    
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
            for point in self.draw_points:
                x, y, z = point['x'], point['y'], point['z']
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
                    self.overlay_data[x, y, z] = 1
                    
            # Update display
            self.update_slice()
            self.status_var.set(f"Drawings loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load drawings: {str(e)}")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = NiftiViewer(root)
    root.mainloop()